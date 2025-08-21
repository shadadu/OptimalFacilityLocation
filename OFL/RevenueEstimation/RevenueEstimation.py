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

def revenue_estimation(lat, lon):
    print(f'Revenue estimation ...')
    # print(f'Revenue Est pop density gee {get_population_density_gee(lat, lon, 500)}')
    # print(f'Fsq count {get_fsq_count(lat, lon, 500)}')
    # print(f'osm poi density {get_osm_poi_density(lat, lon, 500)}')
    # print(f'Get median income {get_median_income_by_point(lat, lon, 500)}')
    return (get_population_density_gee(lat, lon, 500) * 2 +
            get_osm_poi_density(lat, lon, 500) * 100 +
            get_fsq_count(lat, lon, 500) * 50
            )