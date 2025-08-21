import numpy as np
from OFL.Predictors.Categories import get_osm_category, get_foursquare_category
from OFL.Helpers import get_osm_poi_density, get_population_density_gee, get_fips_from_coords, snap_to_nearest_town
from shapely.geometry import Point
from math import radians, cos, sin, asin, sqrt

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

def generate_city_candidate_locations(location_name, radius_c):
    # Use OSMnx to get city polygon
    print(f'Generating candidate locations ...')
    gdf = ox.geocode_to_gdf(location_name)
    city_poly = gdf.geometry.iloc[0]
    bounds = city_poly.bounds

    step = radius_c * 1.5
    deg_step = step / 111_320
    print(f'size of generate city loop: {len(np.arange(bounds[1], bounds[3], deg_step))}')
    candidates = []
    count = 0
    for lat in np.arange(bounds[1], bounds[3], deg_step):
        for lon in np.arange(bounds[0], bounds[2], deg_step):
            p = Point(lon, lat)
            if city_poly.contains(p):
                candidates.append((lat, lon))
        # print(f'current count of generate city loop: {count}')
        count += 1
    return candidates[0:99]


def get_median_income_by_point(lat, lon, radius, CENSUS_API_KEY):
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
    low = 50
    best_radius = 50
    print(f'low, high {low}, {high}')

    # Now generate the actual points with the chosen radius
    step = best_radius * 1.5
    deg_step = step / 111_320
    points = []
    print(f'looper {len(np.arange(center_lat - big_radius/111_320, center_lat + big_radius/111_320, deg_step) )}')
    for lat in np.arange(center_lat - big_radius/111_320,
                         center_lat + big_radius/111_320, deg_step):
        print(f'looper 2 {len(np.arange(center_lon - big_radius/111_320, center_lon + big_radius/111_320, deg_step))}')
        for lon in np.arange(center_lon - big_radius/111_320,
                             center_lon + big_radius/111_320, deg_step):
            # print(f'haversine condition {haversine(center_lat, center_lon, lat, lon)} {big_radius}')
            if haversine(center_lat, center_lon, lat, lon) <= big_radius:
                print(f'haversine condition {haversine(center_lat, center_lon, lat, lon)} {big_radius}')
                points.append((lat, lon))

    print(f"Chosen small_radius: {best_radius:.2f} m, generated {len(points)} subcircles")
    return points[0:99] # , best_radius


def haversine(lon1, lat1, lon2, lat2):
    # Distance in meters
    # print(f'Computing haversine ...')
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


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

from geopy.geocoders import Nominatim
import time

geolocator = Nominatim(user_agent="geo_fallback")





def build_features_for_location(lat, lon, radius_m, cr):
    print(f'Building features for location ...')
    neighborhood_points = generate_circle_points(lat, lon, radius_m, cr)
    print(f'Number of neighborhood points {len(neighborhood_points)}')
    features = []
    for (lat_i, lon_i) in neighborhood_points:
        pop = get_population_density_gee(lat_i, lon_i, cr)
        osm_poi = get_osm_poi_density(lat_i, lon_i, cr)
        fsq_poi = get_fsq_count(lat_i, lon_i, cr)
        # print(f'pop, fsq_poi {pop} {fsq_poi}')
        income = get_median_income_by_point(lat_i, lon_i, cr)
        osm_cat = get_osm_category(lat, lon)
        fsq_cat = get_foursquare_category(lat, lon)
        print(f'Category osm: {osm_cat}, fsq: {fsq_cat}')
        # income = get_median_income_with_radius(lat, lon)
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





