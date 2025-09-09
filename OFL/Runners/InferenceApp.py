import streamlit as st
import requests
import pandas as pd
import joblib
from geopy.geocoders import Nominatim
import ee
from OFL import Helpers
from OFL.Runners.Inference import build_inference_features_for_location
from OFL.Predictors.Predictors import generate_city_candidate_locations, build_features_for_location
import sys

# Global caches for Foursquare(HF + Duckdb)
global _fsq_duckdb_con
def main():
    # -----------------------
    # CONFIG
    # -----------------------
    HF_MODEL_REPO = "shaddie/revenue-predictor"
    HF_MODEL_FILENAME = "model.joblib"  # Model file inside repo

    ee.Authenticate()
    ee.Initialize(project='ee-shaddie77')

    CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # use streamlit secrets to store and retrieve api

    # ------------------------
    # PARAMETERS
    # ------------------------
    radius_m = 100  # Neighborhood radius
    cr = 10  # Subcircle radius
    radius_c = 50  # Candidate facility radius (for city split)
    location_name = "New York, NY"

    # ------------------
    # CACHING
    # ---------------------

    # _fsq_duckdb_con = None
    _fsq_query_cache = {}
    ee.Authenticate()
    ee.Initialize(project='ee-shaddie77')

    _fsq_duckdb_con = None
    _fsq_duckdb_con = Helpers._get_duckdb_connection(_fsq_duckdb_con)

    # --- Parameters

    #
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

    def geocode(geolocation_name):
        geolocator = Nominatim(user_agent="revenue_estimator_app")
        loc = geolocator.geocode(geolocation_name, timeout=10)
        if loc is None:
            raise ValueError(f"Could not geocode location: {geolocation_name}")
        return loc.latitude, loc.longitude

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
            lat, lon = geocode(location_name)
        except Exception as e:
            st.error(f"Geocoding failed: {e}")
            st.stop()

        with st.spinner("Computing Predictors..."):
            X_df = build_inference_features_for_location(lat, lon
                                                         , radius_m
                                                         , cr
                                                         , _fsq_duckdb_con
                                                         , _fsq_query_cache
                                                         , CENSUS_API_KEY)
            # X_df = build_features_for_location(lat, lon
            #                                    , radius_m
            #                                    , cr
            #                                    , _fsq_duckdb_con
            #                                    , _fsq_query_cache
            #                                    , CENSUS_API_KEY)

            """
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
            """
            # Aggregate (by mean) neighborhood features; aggregated for each candidate location
            agg = X_df.mean(numeric_only=True).to_dict()
            st.session_state.locations.append(
                {
                    "location": location_name,
                    "lat": lat,
                    "lon": lon,
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
