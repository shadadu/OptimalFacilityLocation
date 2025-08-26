import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import ee
from OFL.Predictors.Predictors import build_features_for_location, generate_city_candidate_locations
from OFL.Predictors.Categories import encode_location_categories
from OFL.RevenueEstimation.RevenueEstimation import revenue_estimation
from OFL.Runners.CollectRevenueData.RevenueByAssessedValue import revenue_estimation_by_dof_assessment
import time


def build_train_vars(candidates
                     , radius_m
                     , cr
                     , CENSUS_API_KEY
                     , geoclient_app_id
                     , geoclient_app_key
                     ):
    rows = []
    cnt_ = 0
    for lat, lon in candidates:
        X_df = build_features_for_location(lat, lon, radius_m, cr, CENSUS_API_KEY)
        # Aggregate neighborhood features (mean as example)
        X = X_df.mean(numeric_only=True).to_dict()
        # Y = revenue_estimation(lat, lon)
        Y = revenue_estimation_by_dof_assessment(lat, lon
                                                 , geoclient_app_id
                                                 , geoclient_app_key
                                                 , socrata_app_token=None
                                                 , radius_tax_value=200)
        print(f'revenue {Y}')
        X["lat"], X["lon"], X["revenue"] = lat, lon, Y,
        # agg["lat"], agg["lon"] = lat, lon
        rows.append(X)
        cnt_ += 1

    return rows


def build_df(rows, file_path=""):
    df = pd.DataFrame(rows)
    df.to_csv(file_path + 'location_revenue_and_predictors.csv', index=False)
    print(f'Successfully downloaded dataset to specified location')
    pass


def main():
    """
    Collects data from various Geolocation and demographics
     to build data set that is saved to csv for later model training
    """
    ee.Authenticate()
    ee.Initialize(project='ee-shaddie77')

    CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # set via Streamlit secrets or replace string

    # ------------------------
    # PARAMETERS
    # ------------------------
    radius_m = 100  # Neighborhood radius
    cr = 100  # Subcircle radius
    radius_c = 50  # Candidate facility radius (for city split)
    city_name = "New York, NY"

    # --- Parameters
    location_name = "Times Square, New York, NY"
    # location_name = "New York, NY"

    candidates = generate_city_candidate_locations(city_name, radius_c)
    print(f'size of candidates {len(candidates)}')
    print(f'element of candidates {candidates[0]}')

    rows = build_train_vars(candidates, radius_m, cr, CENSUS_API_KEY)
    build_df(rows, "")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    print(f"Execution time: {elapsed_minutes:.2f} minutes")
