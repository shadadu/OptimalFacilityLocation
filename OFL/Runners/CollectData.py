import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import ee
from OFL.Predictors.Predictors import build_features_for_location, generate_city_candidate_locations
from OFL.Predictors.Categories import encode_location_categories
from OFL.RevenueEstimation.RevenueEstimation import revenue_estimation
# from OFL.Runners.CollectRevenueData.RevenueByAssessedValue import revenue_estimation_by_dof_assessment
from OFL.Runners.CollectRevenueData.CollectTaxValueDataNYC import batch_process, query_point, revenue_estimation_by_dof_assessment
import time
import duckdb
import ee

# Global caches
_fsq_duckdb_con = None
_fsq_query_cache = {}

ee.Authenticate()
ee.Initialize(project='ee-shaddie77')

def _get_duckdb_connection():
    """Create and cache a DuckDB connection."""
    global _fsq_duckdb_con
    if _fsq_duckdb_con is None:
        con = duckdb.connect()
        _fsq_duckdb_con = con
    return _fsq_duckdb_con
def build_train_vars(candidates
                     , radius_m
                     , cr
                     , CENSUS_API_KEY
                     , geoclient_subscription_key
                     , SOCRATA_API_KEY
                     ):
    _fsq_duckdb_con = _get_duckdb_connection()
    rows = []
    cnt_ = 0

    for lat, lon in candidates:
        # X_df = build_features_for_location(lat, lon, radius_m, cr, CENSUS_API_KEY)
        # # Aggregate neighborhood features (mean as example)
        # agg = X_df.mean(numeric_only=True).to_dict()
        # # Y = revenue_estimation(lat, lon)
        points = lat, lon
        print(f'Points for tax value {points}')
        Y = query_point(lat, lon)
        # Y = revenue_estimation_by_dof_assessment(lat, lon
        #                                          , geoclient_subscription_key=geoclient_subscription_key
        #                                          , socrata_app_token=SOCRATA_API_KEY
        #                                          , radius_tax_value=200)

        if Y["status"] == "Tax value assigned":
            print(f'revenue Y: {Y}')
            X_df = build_features_for_location(lat, lon, radius_m, cr, _fsq_duckdb_con, _fsq_query_cache, CENSUS_API_KEY)
            # Aggregate neighborhood features (mean as example)
            agg = X_df.mean(numeric_only=True).to_dict()
            # Y = revenue_estimation(lat, lon)
            points = lat, lon
            print(f'Points for tax value {points}')
            agg["lat"], agg["lon"], agg["revenue"] = lat, lon, Y
            rows.append(agg)
        cnt_ += 1

    print(f'Count of successful rows {len(rows)}')
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

    CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # use streamlit secrets to store and retrieve api
    geoclient_key = st.secrets.get("NYC_GEOCLIENT_PRIMARY", "")
    socrata_token = st.secrets.get("SOCRATA_APP_TOKEN", "")

    print(f'secrets geoclient_key: {geoclient_key}, census_api_key: {CENSUS_API_KEY}, socrata_token: {socrata_token}')

    # lat, lon = 40.7128, -74.0060  # Example: Manhattan

    # ------------------------
    # PARAMETERS
    # ------------------------
    radius_m = 100  # Neighborhood radius
    cr = 10  # Subcircle radius
    radius_c = 50  # Candidate facility radius (for city split)
    city_name = "New York, NY"

    # --- Parameters
    location_name = "Times Square, New York, NY"
    # location_name = "New York, NY"

    candidates = generate_city_candidate_locations(city_name, radius_c)
    print(f'size of candidates {len(candidates)}')
    print(f'element of candidates {candidates[0]}')

    rows = build_train_vars(candidates, radius_m, cr, CENSUS_API_KEY, geoclient_key, socrata_token)
    build_df(rows, "/Users/rckyi/Documents/Data/ofl_data.csv")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    print(f"Execution time: {elapsed_minutes:.2f} minutes")
