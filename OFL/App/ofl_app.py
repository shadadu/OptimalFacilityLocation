import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np
from geopy.geocoders import Nominatim

# -----------------------
# CONFIG
# -----------------------
HF_MODEL_REPO = "username/revenue-predictor"  # Change to your HF repo
HF_MODEL_FILENAME = "model.joblib"            # Model file inside repo

# -----------------------
# PLACEHOLDER FUNCTIONS
# -----------------------
def get_population_density_gee(lat, lon, radius_m):
    """Placeholder: compute population density from Google Earth Engine."""
    return np.random.uniform(1000, 10000)  # fake value

def get_osm_poi_density(lat, lon, radius_m):
    """Placeholder: compute POI density from OSM."""
    return np.random.uniform(10, 500)  # fake value

def get_median_income_by_point(lat, lon, radius_m):
    """Placeholder: get median income from Census."""
    return np.random.uniform(30000, 120000)  # fake value

# -----------------------
# UTIL
# -----------------------
@st.cache_resource
def load_model_from_hf():
    """Download model from Hugging Face Hub."""
    url = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{HF_MODEL_FILENAME}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(HF_MODEL_FILENAME, "wb") as f:
        f.write(r.content)
    model = joblib.load(HF_MODEL_FILENAME)
    return model

def geocode(location_name):
    geolocator = Nominatim(user_agent="revenue_estimator_app")
    loc = geolocator.geocode(location_name, timeout=10)
    if loc is None:
        raise ValueError(f"Could not geocode location: {location_name}")
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
        pop_density = get_population_density_gee(lat, lon, radius_m)
        poi_density = get_osm_poi_density(lat, lon, radius_m)
        median_income = get_median_income_by_point(lat, lon, radius_m)

    st.session_state.locations.append({
        "location": location_name,
        "lat": lat,
        "lon": lon,
        "radius_m": radius_m,
        "population_density": pop_density,
        "poi_density": poi_density,
        "median_income": median_income
    })

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
            feature_cols = ["population_density", "poi_density", "median_income"]
            df["estimated_revenue"] = model.predict(df[feature_cols])

        st.subheader("💰 Revenue Estimates")
        st.dataframe(df[["location", "population_density", "poi_density", "median_income", "estimated_revenue"]])

    # Optional: clear button
    if st.button("Clear all locations"):
        st.session_state.locations.clear()
