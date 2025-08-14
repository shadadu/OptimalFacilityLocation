import requests
import osmnx as ox
import numpy as np
import streamlit as st
import duckdb
import requests
import osmnx as ox
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from geopy.geocoders import Nominatim
import altair as alt
from shapely.geometry import Point
import geopandas as gpd
from math import radians, cos, sin, asin, sqrt
import ee
import time, requests
import os
from io import BytesIO
import zipfile
import tempfile


ee.Authenticate()
ee.Initialize(project='ee-shaddie77')

CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # set via Streamlit secrets or replace string
HUGGINGFACE_DATASET = "foursquare/fsq-os-places"
HF_PARQUET_API = f"https://datasets-server.huggingface.co/parquet?dataset={HUGGINGFACE_DATASET}"

# ------------------------
# PARAMETERS
# ------------------------
radius_m = 1000       # Neighborhood radius
cr = 250              # Subcircle radius
radius_c = 500        # Candidate facility radius (for city split)
city_name = "New York, NY"


# --- Parameters
location_name = "Times Square, New York, NY"
# --- Geocode location
lat, lon = ox.geocoder.geocode(location_name)

# ------------------------
# GEE + OSM + FSQ HELPERS
# ------------------------

# FSQ: Load Hugging Face parquet link for fsq-os-places
def get_fsq_count(lat, lon, r):
    print(f'Getting Four Square Count')
    api_url = "https://datasets-server.huggingface.co/parquet?dataset=foursquare/fsq-os-places"
    j = requests.get(api_url).json()
    parquet_urls = [f['url'] for f in j.get('parquet_files', []) if f['split'] == 'train']
    if not parquet_urls:
        return 0
    url = parquet_urls[0]

    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    deg = r / 111_320
    min_lat, max_lat = lat - deg, lat + deg
    min_lon, max_lon = lon - deg, lon + deg

    query = f"""
    SELECT COUNT(*) as count
    FROM '{url}'
    WHERE latitude BETWEEN {min_lat} AND {max_lat}
      AND longitude BETWEEN {min_lon} AND {max_lon}
    """
    res = con.execute(query).df()
    return int(res['count'][0]) if res.shape[0] else 0


# OSM: Count amenities in radius
def get_osm_poi_density(lat, lon, radius):
    print(f'Getting poi density')
    tags = {"amenity": True}
    pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
    return len(pois)

def get_fips_from_coords(lat, lon, retries=3, wait=5):
    url = "https://geo.fcc.gov/api/census/block/find"
    params = {"latitude": lat, "longitude": lon, "format": "json"}
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 502 and i < retries - 1:
                time.sleep(wait)
                continue
            raise

# Placeholder for median income retrieval
def get_median_income_by_point(lat, lon, radius):
    # TODO: Replace with buffered multi-tract ACS query
    print(f'Getting media_income ...')
    """Use FCC API to find block FIPS then Census ACS to fetch B19013_001E (median household income)."""
    if not CENSUS_API_KEY:
        raise RuntimeError("CENSUS_API_KEY is not set. Put your key in Streamlit secrets or set variable.")
    # FCC to get block FIPS
    j = get_fips_from_coords(lat, lon, retries=3, wait=5)
    block_fips = j.get("Block", {}).get("FIPS")
    if not block_fips:
        return None
    state_fips = block_fips[0:2]
    county_fips = block_fips[2:5]
    tract_fips = block_fips[5:11]

    headers = {
        "X-API-Key": CENSUS_API_KEY
    }

    acs_url = (
        "https://api.census.gov/data/2022/acs/acs5"
        f"?get=B19013_001E&for=tract:{tract_fips}&in=state:{state_fips}%20county:{county_fips}&key={CENSUS_API_KEY}"
    )
    r2 = requests.get(acs_url, headers=headers, timeout=50)
    r2.raise_for_status()
    arr = r2.json()
    if len(arr) < 2:
        return None
    val = arr[1][0]
    try:
        return float(val) if val not in (None, "", "null") else None
    except Exception:
        return None

def load_tracts_from_hf(hf_url):
    # Download ZIP into memory
    r = requests.get(hf_url, timeout=30)
    r.raise_for_status()
    z = BytesIO(r.content)

    # Extract to a temporary folder
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(z) as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find the .shp file inside the extracted contents
        shp_file = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]

        # Read shapefile into GeoDataFrame
        tracts = gpd.read_file(shp_file).to_crs(epsg=4326)

    return tracts

def get_median_income_with_radius(lat, lon, radius_m=1000, census_api_key=CENSUS_API_KEY
                                  , hf_url ="https://huggingface.co/datasets/shaddie/tiger-tracts/resolve/main/tl_2024_01_tract.zip"
                                  ):
    """
    Calculates the weighted median income within a radius by:
      1. Loading U.S. Census tract boundaries from a Hugging Face-hosted ZIP (in-memory).
      2. Buffering a point and selecting intersecting tracts.
      3. Fetching median-income and household counts via ACS API.
      4. Returning weighted average income.
    """
    if not census_api_key:
        raise ValueError("Census API key required")

    tracts = load_tracts_from_hf(hf_url)

    # 2. Create spatial buffer
    buffer_deg = radius_m / 111_320.0
    buffer_geom = Point(lon, lat).buffer(buffer_deg)
    tracts_sel = tracts[tracts.geometry.intersects(buffer_geom)]
    if tracts_sel.empty:
        return None

    # 3. Query ACS for each tract's median income & household count
    incomes = []
    weights = []
    year = hf_url.split("_")[-2] if "2024" in hf_url else "2024"
    for _, row in tracts_sel.iterrows():
        s = row["STATEFP"]
        c = row["COUNTYFP"]
        t = row["TRACTCE"]
        acs_url = (
            f"https://api.census.gov/data/{year}/acs/acs5"
            f"?get=B19013_001E,B11016_001E"
            f"&for=tract:{t}&in=state:{s}%20county:{c}&key={census_api_key}"
        )
        r2 = requests.get(acs_url, timeout=30)
        if r2.status_code != 200:
            continue
        arr = r2.json()
        if len(arr) < 2 or any(val in (None, "", "null") for val in arr[1]):
            continue
        inc, hh = arr[1]
        incomes.append(float(inc))
        weights.append(float(hh))

    if not incomes:
        return None

    # 4. Weighted average
    return sum(i * w for i, w in zip(incomes, weights)) / sum(weights)


# Population density via GEE placeholder
def get_population_density_gee(lat, lon, radius_m):
    # TODO: Replace with actual Earth Engine query
    print(f'Getting population density gee')
    dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filter(ee.Filter.date('2020-01-01', '2020-12-31')) \
        .first()
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(radius_m).bounds()
    stats = dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=100,
        maxPixels=1e9
    )
    ans = stats.getInfo().get('population', 0)
    return 0 if ans == None else 0


# ------------------------
# GEOMETRY HELPERS
# ------------------------

def haversine(lon1, lat1, lon2, lat2):
    # Distance in meters
    print(f'Computing haversine ...')
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

def generate_circle_points(center_lat, center_lon, big_radius, small_radius):
    # Generates subcircle centers within big circle
    print(f'Generating circle points')
    points = []
    step = small_radius * 1.5
    deg_step = step / 111_320
    for lat in np.arange(center_lat - big_radius/111_320,
                         center_lat + big_radius/111_320, deg_step):
        for lon in np.arange(center_lon - big_radius/111_320,
                             center_lon + big_radius/111_320, deg_step):
            if haversine(center_lon, center_lat, lon, lat) <= big_radius:
                points.append((lat, lon))
    return points


# ------------------------
# FEATURE BUILDER
# ------------------------

def build_features_for_location(lat, lon, radius_m, cr):
    print(f'Building features for location ...')
    neighborhood_points = generate_circle_points(lat, lon, radius_m, cr)
    features = []
    for (lat_i, lon_i) in neighborhood_points:
        pop = get_population_density_gee(lat_i, lon_i, cr)
        # osm_poi = get_osm_poi_density(lat_i, lon_i, cr)
        fsq_poi = get_fsq_count(lat_i, lon_i, cr)
        # income = get_median_income_by_point(lat_i, lon_i, cr)
        income = get_median_income_with_radius(lat, lon)
        features.append({
            "lat": lat_i,
            "lon": lon_i,
            "population_density": pop,
            # "osm_poi_density": osm_poi,
            "fsq_poi_count": fsq_poi,
            # "median_income": income
        })
    return pd.DataFrame(features)


# ------------------------
# CITY-WIDE CANDIDATE LOCATIONS
# ------------------------

def generate_city_candidate_locations(city_name, radius_c):
    # Use OSMnx to get city polygon
    print(f'Generating candidate locations ...')
    gdf = ox.geocode_to_gdf(city_name)
    city_poly = gdf.geometry.iloc[0]
    bounds = city_poly.bounds

    step = radius_c * 1.5
    deg_step = step / 111_320
    candidates = []
    for lat in np.arange(bounds[1], bounds[3], deg_step):
        for lon in np.arange(bounds[0], bounds[2], deg_step):
            p = Point(lon, lat)
            if city_poly.contains(p):
                candidates.append((lat, lon))
    return candidates


# ------------------------
# MAIN PIPELINE
# ------------------------

def revenue_estimation(lat, lon):
    # TODO: Replace with your real revenue estimation logic
    print(f'Revenue estimation ...')
    return (get_population_density_gee(lat, lon, 500) * 2 +
            # get_osm_poi_density(lat, lon, 500) * 100 +
            get_fsq_count(lat, lon, 500) * 50 #+
            # get_median_income_by_point(lat, lon, 500) * 0.01
            )

# Step 3: Get candidate facility locations
candidates = generate_city_candidate_locations(city_name, radius_c)

# Step 4 & 5: Build dataset and run regression
rows = []
for lat, lon in candidates:
    X_df = build_features_for_location(lat, lon, radius_m, cr)
    # Aggregate neighborhood features (mean as example)
    agg = X_df.mean(numeric_only=True).to_dict()
    Y = revenue_estimation(lat, lon)
    agg["lat"], agg["lon"], agg["revenue"] = lat, lon, Y
    rows.append(agg)

df = pd.DataFrame(rows)

# Fit regression
X = df[["population_density", "osm_poi_density", "fsq_poi_count", "median_income"]]
y = df["revenue"]
model = LinearRegression().fit(X, y)

print("Regression coefficients:", model.coef_)
print("Intercept:", model.intercept_)
