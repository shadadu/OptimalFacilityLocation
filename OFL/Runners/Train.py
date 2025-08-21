import streamlit as st
import osmnx as ox
import pandas as pd
from sklearn.linear_model import LinearRegression
import ee
from OFL.Predictors.Predictors import build_features_for_location, generate_city_candidate_locations
from OFL.Predictors.Categories import encode_location_categories
from OFL.RevenueEstimation.RevenueEstimation import revenue_estimation


def build_train_vars(candidates, radius_m, cr):
    rows = []
    cnt_ = 0
    for lat, lon in candidates:
        # print(f'cnt_ {cnt_}')
        X_df = build_features_for_location(lat, lon, radius_m, cr)
        # Aggregate neighborhood features (mean as example)
        agg = X_df.mean(numeric_only=True).to_dict()
        Y = revenue_estimation
        print(f'revenue {Y}')
        agg["lat"], agg["lon"], agg["revenue"] = lat, lon, Y,
        # agg["lat"], agg["lon"] = lat, lon
        rows.append(agg)
        cnt_ += 1

    return rows


def build_xy(rows):
    df = pd.DataFrame(rows)
    df_vars = encode_location_categories(df)
    X = df_vars[["population_density"
        , "osm_poi_density"
        , "fsq_poi_count"
        , "median_income"
        , "fsq_category_encoded"
        , "osm_category_encoded"]]
    y = df_vars["revenue"]

    return X, y


def train(X, y, model):
    if model is None:
        model = LinearRegression().fit(X, y)
        print("Regression coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        return model
    else:
        return None


def main():
    """
    This function contains the main logic of the script.
    """
    ee.Authenticate()
    ee.Initialize(project='ee-shaddie77')

    CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # set via Streamlit secrets or replace string
    HUGGINGFACE_DATASET = "foursquare/fsq-os-places"
    HF_PARQUET_API = f"https://datasets-server.huggingface.co/parquet?dataset={HUGGINGFACE_DATASET}"

    # ------------------------
    # PARAMETERS
    # ------------------------
    radius_m = 100  # Neighborhood radius
    cr = 50  # Subcircle radius
    radius_c = 50  # Candidate facility radius (for city split)
    city_name = "New York, NY"

    # --- Parameters
    location_name = "Times Square, New York, NY"
    # location_name = "New York, NY"
    # --- Geocode location
    lat, lon = ox.geocoder.geocode(city_name)

    candidates = generate_city_candidate_locations(city_name, radius_c)
    print(f'size of candidates {len(candidates)}')
    print(f'element of candidates {candidates[0]}')

    rows = build_train_vars(candidates, radius_m, cr)

    X, y = build_xy(rows)

    model = train(X, y, None)


if __name__ == "__main__":
    main()
