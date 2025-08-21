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


def get_nearest_place_coords(lat, lon):
    """
    Returns (lat, lon) of nearest city/town/village center to the input coordinates.
    Uses Nominatim reverse + forward geocoding.
    """
    geolocator = Nominatim(user_agent="geo-fallback-app")

    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and "address" in location.raw:
            addr = location.raw["address"]
            place = addr.get("city") or addr.get("town") or addr.get("village")
            if place:
                place_loc = geolocator.geocode(place)
                if place_loc:
                    return (place_loc.latitude, place_loc.longitude)
    except Exception as e:
        print(f"Geocoding fallback failed: {e}")

    return None

# ----------------------------
# POI Density (with fallback)
# ----------------------------
def get_osm_poi_density(lat, lon, radius, max_expand=3, expand_factor=2):
    """
    Get POI density from OSM.
    Expands radius if no POIs found, and falls back to nearest
    town/city center if still empty.
    """
    print(f"Getting POI density at ({lat}, {lon}), radius={radius}m")

    attempt_radius = radius
    tags = {"amenity": True}

    # Try with expanding radius
    for attempt in range(max_expand + 1):
        try:
            pois = ox.features_from_point((lat, lon), tags=tags, dist=attempt_radius)
            if len(pois) > 0:
                print(f'Found osm_pois {len(pois)}')
                return len(pois)
            else:
                print(f"No POIs at radius {attempt_radius}m, expanding search...")
        except Exception as e:
            print(f"OSM query failed at radius {attempt_radius}m: {e}")

        attempt_radius *= expand_factor

    # Fallback to nearest city/town
    print("No POIs found after expansions. Falling back to nearest town/city center...")
    fallback_coords = get_nearest_place_coords(lat, lon)
    if fallback_coords:
        try:
            pois = ox.features_from_point(fallback_coords, tags=tags, dist=radius)
            if len(pois) > 0:
                print(f'Found osm poi {len(pois)}')
                return len(pois)
        except Exception as e:
            print(f"OSM fallback query failed: {e}")

    print("No POIs found, even after fallback.")
    return 0
def get_population_density_gee(lat, lon, radius_m, max_expand=3, expand_factor=2):
    """
    Get population density from WorldPop using Earth Engine.
    Expands radius if no values are found, and falls back to nearest
    city/town center if still empty.
    """
    print(f"Getting population density at ({lat}, {lon}), radius={radius_m}m")

    dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filter(ee.Filter.date('2020-01-01', '2020-12-31')) \
        .first()

    attempt_radius = radius_m

    # Try with expanding radius
    for attempt in range(max_expand + 1):
        try:
            point = ee.Geometry.Point(lon, lat)
            region = point.buffer(attempt_radius).bounds()
            stats = dataset.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            )
            result = stats.getInfo()
            pop_val = result.get('population', None)
            if pop_val is not None:
                print(f'Found pop_val {pop_val}')
                return pop_val
            else:
                print(f"No population data at radius {attempt_radius}m, expanding search...")
        except Exception as e:
            print(f"GEE query failed at radius {attempt_radius}m: {e}")

        attempt_radius *= expand_factor

    # Fallback to nearest city/town
    print("No population found after expansions. Falling back to nearest town/city center...")
    fallback_coords = get_nearest_place_coords(lat, lon)
    if fallback_coords:
        try:
            point = ee.Geometry.Point(fallback_coords[::-1])  # (lon, lat)
            region = point.buffer(radius_m).bounds()
            stats = dataset.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            )
            result = stats.getInfo()
            pop_val = result.get('population', None)
            if pop_val is not None:
                print(f'Found pop_val {pop_val}')
                return pop_val
        except Exception as e:
            print(f"GEE fallback query failed: {e}")

    print("No population data found, even after fallback.")
    return 0

def snap_to_nearest_town(lat, lon):
    """
    Snap (lat, lon) to nearest town/city center if available.
    """
    geolocator = Nominatim(user_agent="geo_fallback")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location and "town" in location.raw["address"]:
            town = location.raw["address"]["town"]
        elif location and "city" in location.raw["address"]:
            town = location.raw["address"]["city"]
        else:
            return lat, lon  # no town/city info, return same coords

        # Forward geocode the town name → town center coords
        town_loc = geolocator.geocode(town)
        if town_loc:
            return town_loc.latitude, town_loc.longitude
    except Exception as e:
        print(f"Town fallback failed: {e}")
    return lat, lon  # fallback to original point

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