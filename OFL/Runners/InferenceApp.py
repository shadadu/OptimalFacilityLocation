import streamlit as st
import requests, certifi
import hashlib
import json
import os
import time
import pandas as pd
import joblib
import ee
from OFL import Helpers
from OFL.Runners.Inference import build_inference_features_for_location
from OFL.Predictors.Predictors import generate_city_candidate_locations

# Global caches for Foursquare(HF + Duckdb)
global _fsq_duckdb_con


def main():
    print(f'Starting the app... ')
    # -----------------------
    # CONFIG
    # -----------------------
    HF_MODEL_REPO = "shaddie/revenue-predictor"
    HF_MODEL_FILENAME = "model.joblib"  # Model file inside repo

    CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # use streamlit secrets to store and retrieve api
    print(f'Obtained Census Api key ...')

    print(f'Setting up EE connection...')
    ee.Authenticate()
    ee.Initialize(project='ee-shaddie77')
    print(f'Google EE con initialized ...')

    # ------------------------
    # PARAMETERS
    # ------------------------
    radius_m = 100  # Neighborhood radius
    cr = 10  # Subcircle radius
    radius_c = 50  # Candidate facility radius (for city split)
    location_name = "New York, NY"

    # ------------------
    # CACHING
    # --------------------
    _GEOCODE_CACHE_FILE = "/Users/rckyi/Documents/Data/geocode_cache.json"
    _geocode_cache = {}

    # Load geocode cache if available
    if os.path.exists(_GEOCODE_CACHE_FILE):
        try:
            with open(_GEOCODE_CACHE_FILE, "r") as f:
                _geocode_cache = json.load(f)
        except Exception:
            _geocode_cache = {}

    print(f'Connecting to Foursquare db: DuckDB + HF... ')
    # Foursquare caching
    _fsq_query_cache = {}
    _fsq_duckdb_con = None
    _fsq_duckdb_con = Helpers._get_duckdb_connection(_fsq_duckdb_con)

    print(f'Foursquare db con established ...')

    # --- Parameters

    print(f'Getting candidates for the location')
    candidates = generate_city_candidate_locations(location_name, radius_c)
    print(f'size of candidates {len(candidates)}')
    print(f'element of candidates {candidates[0]}')

    @st.cache_resource
    def load_model_from_hf():
        """Download model from Hugging Face Hub."""
        url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{HF_MODEL_FILENAME}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(HF_MODEL_FILENAME, "wb") as f:
            f.write(r.content)
        model_hf = joblib.load(HF_MODEL_FILENAME)
        return model_hf

    def _save_cache():
        """Persist cache to disk"""
        with open(_GEOCODE_CACHE_FILE, "w") as f:
            json.dump(_geocode_cache, f)

    def geocode_direct(geolocation_name, use_cache=True, rate_limit=1.0):
        """
        Geocode a place name into (lat, lon).
        Uses direct Nominatim API via requests (with certifi).
        Caches results on disk to avoid repeated hits.
        Respects a simple rate limit (default: 1 second).
        """
        key = hashlib.sha1(geolocation_name.strip().lower().encode()).hexdigest()

        # ✅ Cache check
        if use_cache and key in _geocode_cache:
            return _geocode_cache[key]

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": geolocation_name,
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "revenue_estimator_app"}

        try:
            # ✅ Rate limit: sleep before request
            if rate_limit > 0:
                time.sleep(rate_limit)

            resp = requests.get(url, params=params, headers=headers, verify=certifi.where(), timeout=10)
            resp.raise_for_status()
            results = resp.json()
            if not results:
                raise ValueError(f"Could not geocode location: {geolocation_name}")
            latlon = (float(results[0]["lat"]), float(results[0]["lon"]))

            # ✅ Cache result
            if use_cache:
                _geocode_cache[key] = latlon
                _save_cache()

            return latlon
        except Exception as e:
            raise RuntimeError(f"Geocoding failed for {geolocation_name}: {e}")

    # -----------------------
    # STREAMLIT UI
    # -----------------------
    st.set_page_config(layout="wide", page_title="Revenue Estimator (Inference)")

    st.title("📍 Revenue Estimator (Inference)")

    if "locations" not in st.session_state:
        st.session_state.locations = []

    # --- Form to add locations ---
    with st.form("add_loc"):
        location_name = st.text_input("Location name / address", "Times Square, New York, NY")
        radius_m = st.number_input("Radius (meters)", min_value=100, max_value=5000, value=500, step=50)
        submitted = st.form_submit_button("Add location")

    if submitted:
        try:
            lat, lon = geocode_direct(location_name)
            print(f'Geocoded lat, lon {lat}, {lon}')
        except Exception as e:
            st.error(f"Geocoding failed: {e}")
            st.stop()
        print(f'compute predictors ...')
        with st.spinner("Computing Predictors..."):
            X_df = build_inference_features_for_location(lat, lon
                                                         , radius_m
                                                         , cr
                                                         , _fsq_duckdb_con
                                                         , _fsq_query_cache
                                                         , CENSUS_API_KEY)

            # Aggregate (by mean) neighborhood features; aggregated for each candidate location
            agg = X_df.mean(numeric_only=True).to_dict()
            st.session_state.locations.append(
                {
                    "location": location_name,
                    "lat": lat,
                    "lon": lon,
                    "radius": radius_m,
                    "population_density": agg["population_density"],
                    "osm_poi_density": agg["osm_poi_density"],
                    "median_income": agg["median_income"]
                }  # tbd: add categories data
            )

    # --- Display locations in a table ---
    if st.session_state.locations:
        df = pd.DataFrame(st.session_state.locations)
        st.subheader("📋 Added Locations")
        st.dataframe(df)

    # --- Button to run inference ---
    if st.button("Run inference"):
        with st.spinner("Loading pretrained model..."):
            model = load_model_from_hf()

        with st.spinner("Running inference..."):
            feature_cols = ["population_density", "poi_density", "median_income"]  # tbd: update feature columns
            df["estimated_revenue"] = model.predict(df[feature_cols])

        st.subheader("💰 Revenue Estimates")
        st.dataframe(df[["location", "population_density", "poi_density", "median_income", "estimated_revenue"]])

    # Optional: clear button
    if st.button("Clear all locations"):
        st.session_state.locations.clear()


# Call the main function
if __name__ == "__main__":
    main()
