import numpy as np
from OFL.Predictors.Categories import get_osm_category, get_foursquare_category
from OFL.Helpers import get_osm_poi_density, get_population_density_gee, get_fips_from_coords, snap_to_nearest_town
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt
import time, requests
import os
import hashlib


def generate_city_candidate_locations(location_name, radius_c):
    # Use OSMnx to get city polygon
    print(f'Generating candidate locations ...')
    gdf = ox.geocode_to_gdf(location_name)
    city_poly = gdf.geometry.iloc[0]
    bounds = city_poly.bounds

    print(f'City bounds {bounds}')

    step = radius_c * 1.5 * 50
    deg_step = step / 111_320

    print(f'size of generate city loop for deg_step {deg_step}: {len(np.arange(bounds[1], bounds[3], deg_step))}')
    candidates = []
    count = 0
    for lat in np.arange(bounds[1], bounds[3], deg_step):
        for lon in np.arange(bounds[0], bounds[2], deg_step):
            # Create a Shapely Point object
            p = Point(lon, lat)
            if city_poly.contains(p):
                candidates.append((lat, lon))
        # print(f'current count of generate city loop: {count}')
        count += 1
    # print(f'Candidates {candidates}')
    return candidates


def get_median_income_by_point(lat, lon, radius, CENSUS_API_KEY):
    # TODO: Replace with buffered multi-tract ACS query
    print(f'Getting median_income ...')
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
        print(f'Unable to find median income')
        return None
    val = arr[1][0]
    print(f'median income value {val}')
    try:
        return float(val) if val not in (None, "", "null") else None
    except Exception:
        return None

def generate_circle_points(center_lat, center_lon, big_radius, N=10):
    """
    Generates subcircle centers within big circle.
    small_radius is chosen so that the number of subcircles <= N.
    """
    print(f'Generating circle points with max {N} subcircles')

    def count_points_for_radius(small_radius):
        step = small_radius * 1.5
        deg_step = step / 111_320
        count = 0
        for lat in np.arange(center_lat - big_radius/111_320,
                             center_lat + big_radius/111_320, deg_step):
            for lon in np.arange(center_lon - big_radius/111_320,
                                 center_lon + big_radius/111_320, deg_step):
                if haversine(center_lon, center_lat, lon, lat) <= big_radius:
                    count += 1
        return count

    # Binary search for the largest small_radius that satisfies count <= N
    low, high = 1.0, big_radius  # meters
    best_radius = low
    for _ in range(30):  # enough iterations for sub-meter precision
        mid = (low + high) / 2
        if count_points_for_radius(mid) <= N:
            best_radius = mid
            low = mid
        else:
            high = mid
    # low = 50
    # best_radius = 50
    print(f'low, high {low}, {high}')

    # Now generate the actual points with the chosen radius
    step = best_radius * 1.5
    deg_step = step / 111_320
    points = []
    for lat in np.arange(center_lat - big_radius/111_320,
                         center_lat + big_radius/111_320, deg_step):
        for lon in np.arange(center_lon - big_radius/111_320,
                             center_lon + big_radius/111_320, deg_step):
            if haversine(center_lat, center_lon, lat, lon) <= big_radius:
                points.append((lat, lon))

    print(f"Chosen small_radius: {best_radius:.2f} m, generated {len(points)} subcircles")
    return points # , best_radius


def haversine(lon1, lat1, lon2, lat2):
    # Distance in meters
    # print(f'Computing haversine ...')
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))



def _ensure_local_parquet():
    """
    Ensure the Foursquare parquet file exists locally.
    Downloads once from Hugging Face if not already present.
    """
    # Paths
    FSQ_LOCAL_FILE = "/Users/rckyi/Documents/Data/fsq_places.parquet"
    FSQ_DATASET_META = "https://datasets-server.huggingface.co/parquet?dataset=foursquare/fsq-os-places"

    if os.path.exists(FSQ_LOCAL_FILE):
        return FSQ_LOCAL_FILE

    print("📥 Downloading FSQ parquet file from Hugging Face...")

    # Step 1: Fetch metadata once
    j = requests.get(FSQ_DATASET_META, timeout=15).json()
    parquet_urls = [f['url'] for f in j.get('parquet_files', []) if f['split'] == 'train']
    if not parquet_urls:
        raise RuntimeError("No parquet URLs found for Foursquare dataset")
    remote_url = parquet_urls[0]

    # Step 2: Stream download
    with requests.get(remote_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(FSQ_LOCAL_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"✅ Saved FSQ parquet to {FSQ_LOCAL_FILE}")
    return FSQ_LOCAL_FILE



def get_fsq_count(lat, lon, r,  _fsq_query_cache, _fsq_duckdb_con):
    """
    Count FSQ places within radius r (meters) of lat/lon.
    Uses local parquet + cache to avoid repeated requests.
    """
    # Cache key
    key = hashlib.md5(f"{lat:.6f}_{lon:.6f}_{r}".encode()).hexdigest()
    if key in _fsq_query_cache:
        return _fsq_query_cache[key]

    print("Getting Foursquare Count")

    local_file = _ensure_local_parquet()
    con = _fsq_duckdb_con
    # con = _get_duckdb_connection()

    # Convert radius meters to degrees (approx)
    deg = r / 111_320
    min_lat, max_lat = lat - deg, lat + deg
    min_lon, max_lon = lon - deg, lon + deg

    query = f"""
    SELECT COUNT(*) as count
    FROM read_parquet('{local_file}')
    WHERE latitude BETWEEN {min_lat} AND {max_lat}
      AND longitude BETWEEN {min_lon} AND {max_lon}
    """

    res = con.execute(query).fetchdf()
    count = int(res['count'][0]) if res.shape[0] else 0

    _fsq_query_cache[key] = count
    return count



def category_with_fallback(lat, lon, fetch_fn, radii=[200, 500, 1000, 2000], delay=1):
    """
    Try to fetch category with expanding radius.
    If still empty, snap to nearest town and retry once.

    fetch_fn: function(lat, lon, radius) -> str | None
    """
    for r in radii:
        try:
            category = fetch_fn(lat, lon, r)
            if category:  # got something
                return category
        except Exception as e:
            print(f"Fetch attempt failed at radius {r}: {e}")
        time.sleep(delay)

    # Snap to nearest town & retry once
    town_lat, town_lon = snap_to_nearest_town(lat, lon)
    if (town_lat, town_lon) != (lat, lon):
        return category_with_fallback(town_lat, town_lon, fetch_fn, radii, delay)

    return "Unknown"


def build_features_for_location(lat, lon, radius_m, cr, _fsq_duckdb_con, _fsq_query_cache, CENSUS_API_KEY):
    print(f'Building features for location ...')
    neighborhood_points = generate_circle_points(lat, lon, radius_m, cr)
    print(f'Number of neighborhood points {len(neighborhood_points)}')
    features = []
    for (lat_i, lon_i) in neighborhood_points:
        pop = get_population_density_gee(lat_i, lon_i, cr)
        osm_poi = get_osm_poi_density(lat_i, lon_i, cr)
        fsq_poi = get_fsq_count(lat_i, lon_i, cr, _fsq_duckdb_con, _fsq_query_cache)
        income = get_median_income_by_point(lat_i, lon_i, cr, CENSUS_API_KEY)
        osm_cat = get_osm_category(lat, lon)
        fsq_cat = get_foursquare_category(lat, lon)
        features.append({
            "lat": lat_i,
            "lon": lon_i,
            "population_density": pop,
            "osm_poi_density": osm_poi,
            "fsq_poi_count": fsq_poi,
            "median_income": income,
            "location_category_foursquare": fsq_cat,
            "location_category_osm": osm_cat,
        })
    return pd.DataFrame(features)





