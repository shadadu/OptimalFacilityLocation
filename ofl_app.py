"""
Streamlit app: Revenue estimator + live regression
Uses: DuckDB + HuggingFace parquet (foursquare/fsq-os-places) for POIs (no full download)
      U.S. Census (ACS) for median income (tract)
"""

import streamlit as st
import requests
import duckdb
import pandas as pd
import statsmodels.api as sm
from geopy.geocoders import Nominatim
import altair as alt

# -----------------------
# CONFIG (replace with your API key)
# -----------------------
CENSUS_API_KEY = st.secrets.get("CENSUS_API_KEY", "")  # set via Streamlit secrets or replace string
HUGGINGFACE_DATASET = "foursquare/fsq-os-places"
HF_PARQUET_API = f"https://datasets-server.huggingface.co/parquet?dataset={HUGGINGFACE_DATASET}"

# -----------------------
# UTIL: DuckDB connection & parquet URL discovery (cached)
# -----------------------
@st.cache_resource
def get_duckdb_conn():
    con = duckdb.connect(database=":memory:")
    # enable httpfs to read parquet over https
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    return con

@st.cache_data(show_spinner=False)
def get_parquet_url():
    """Query HF datasets-server to get a parquet file URL for the dataset."""
    resp = requests.get(HF_PARQUET_API, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    parquet_files = j.get("parquet_files", []) or j.get("parquets", []) or []
    # choose first URL that looks like a parquet file
    urls = []
    for f in parquet_files:
        url = f.get("url") or f.get("file")
        if not url:
            continue
        if url.endswith(".parquet") or "parquet" in url:
            urls.append(url)
    if not urls:
        raise RuntimeError("No parquet URL found for dataset via Hugging Face datasets-server API.")
    return urls[0]

# -----------------------
# UTIL: detect lat/lon column names from parquet schema
# -----------------------
@st.cache_data
def detect_latlon_columns(parquet_url):
    con = get_duckdb_conn()
    # read zero rows to get column names
    try:
        zero_df = con.execute(f"SELECT * FROM '{parquet_url}' LIMIT 0").df()
    except Exception as e:
        raise RuntimeError(f"Failed to inspect parquet schema: {e}")
    cols_lower = [c.lower() for c in zero_df.columns]
    # common names
    if "latitude" in cols_lower and "longitude" in cols_lower:
        lat_col = zero_df.columns[cols_lower.index("latitude")]
        lon_col = zero_df.columns[cols_lower.index("longitude")]
    elif "lat" in cols_lower and "lon" in cols_lower:
        lat_col = zero_df.columns[cols_lower.index("lat")]
        lon_col = zero_df.columns[cols_lower.index("lon")]
    elif "lat" in cols_lower and "longitude" in cols_lower:
        lat_col = zero_df.columns[cols_lower.index("lat")]
        lon_col = zero_df.columns[cols_lower.index("longitude")]
    else:
        # fallback: try to find any pair-like names
        candidates = [c for c in zero_df.columns if c.lower().startswith(("lat", "lon", "long", "latitude", "longitude"))]
        if len(candidates) >= 2:
            # pick first two that look reasonable
            lat_col, lon_col = candidates[0], candidates[1]
        else:
            raise RuntimeError(f"Could not auto-detect lat/lon columns. Parquet columns: {zero_df.columns.tolist()}")
    return lat_col, lon_col

# -----------------------
# Query functions
# -----------------------
def geocode(location_name):
    geolocator = Nominatim(user_agent="revenue_estimator_app")
    loc = geolocator.geocode(location_name, timeout=10)
    if loc is None:
        raise ValueError(f"Could not geocode location: {location_name}")
    return loc.latitude, loc.longitude

def query_foursquare_bbox(lat, lon, radius_m, sample_limit=300):
    """
    Query the remote parquet via DuckDB for a bounding box around lat/lon.
    Returns: (count, sample_dataframe)
    """
    parquet_url = get_parquet_url()
    con = get_duckdb_conn()
    lat_col, lon_col = detect_latlon_columns(parquet_url)

    # approximate degrees for given radius (good for small radii)
    deg_buffer = radius_m / 111_320.0
    min_lat, max_lat = lat - deg_buffer, lat + deg_buffer
    min_lon, max_lon = lon - deg_buffer, lon + deg_buffer

    # count query
    count_q = f"""
        SELECT COUNT(*) AS cnt
        FROM '{parquet_url}'
        WHERE "{lat_col}" BETWEEN {min_lat} AND {max_lat}
          AND "{lon_col}" BETWEEN {min_lon} AND {max_lon}
    """
    try:
        cnt_df = con.execute(count_q).df()
        count = int(cnt_df["cnt"].iloc[0])
    except Exception as e:
        raise RuntimeError(f"Failed to run count query on parquet: {e}")

    # sample query (fetches small sample to display)
    sample_q = f"""
        SELECT *
        FROM '{parquet_url}'
        WHERE "{lat_col}" BETWEEN {min_lat} AND {max_lat}
          AND "{lon_col}" BETWEEN {min_lon} AND {max_lon}
        LIMIT {sample_limit}
    """
    try:
        sample_df = con.execute(sample_q).df()
    except Exception as e:
        # return empty sample if query fails
        sample_df = pd.DataFrame()
    return count, sample_df

# -----------------------
# Census median income (by point -> tract)
# -----------------------
@st.cache_data
def get_median_income_by_point(lat, lon):
    """Use FCC API to find block FIPS then Census ACS to fetch B19013_001E (median household income)."""
    if not CENSUS_API_KEY:
        raise RuntimeError("CENSUS_API_KEY is not set. Put your key in Streamlit secrets or set variable.")
    # FCC to get block FIPS
    fcc_url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
    r = requests.get(fcc_url, timeout=10)
    r.raise_for_status()
    j = r.json()
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
    r2 = requests.get(acs_url, headers=headers, timeout=10)
    r2.raise_for_status()
    arr = r2.json()
    if len(arr) < 2:
        return None
    val = arr[1][0]
    try:
        return float(val) if val not in (None, "", "null") else None
    except Exception:
        return None

# -----------------------
# Revenue proxy & regression helpers
# -----------------------
def estimate_revenue_from_count(num_pois, revenue_per_poi=1000.0):
    """Simple proxy: multiply POI count by per-POI revenue."""
    return float(num_pois) * float(revenue_per_poi)

def run_regression(df, predictors):
    """Run OLS with predictors list (predictor names must be columns in df)"""
    if df.shape[0] < 2:
        return None
    df_clean = df.dropna(subset=predictors + ["revenue"])
    if df_clean.shape[0] < 2:
        return None
    X = df_clean[predictors]
    X = sm.add_constant(X)
    y = df_clean["revenue"]
    model = sm.OLS(y, X).fit()
    return model

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(layout="wide", page_title="Revenue Estimator (DuckDB + HF)")

st.title("📍 Revenue Estimator (Foursquare via DuckDB + Hugging Face)")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Add a location")
    with st.form("add_loc"):
        location_name = st.text_input("Location name / address", "Times Square, New York, NY")
        radius_m = st.number_input("Radius (meters)", min_value=100, max_value=5000, value=500, step=50)
        revenue_per_poi = st.number_input("Revenue per POI (USD, proxy)", min_value=0.0, value=1000.0, step=100.0)
        submitted = st.form_submit_button("Add location")

    if submitted:
        try:
            lat, lon = geocode(location_name)
        except Exception as e:
            st.error(f"Geocoding failed: {e}")
            st.stop()

        with st.spinner("Querying Foursquare parquet via DuckDB..."):
            try:
                fsq_count, fsq_sample = query_foursquare_bbox(lat, lon, radius_m)
            except Exception as e:
                st.error(f"Foursquare query failed: {e}")
                st.stop()

        with st.spinner("Querying median income (Census)..."):
            try:
                median_income = get_median_income_by_point(lat, lon)
            except Exception as e:
                st.error(f"Census query failed: {e}")
                median_income = None

        revenue_est = estimate_revenue_from_count(fsq_count, revenue_per_poi)

        # append to session
        if "locations" not in st.session_state:
            st.session_state.locations = []
        st.session_state.locations.append({
            "location": location_name,
            "lat": lat,
            "lon": lon,
            "radius_m": radius_m,
            "fsq_count": fsq_count,
            "revenue": revenue_est,
            "median_income": median_income
        })

        # show sample (small)
        if not fsq_sample.empty:
            st.subheader("Sample POIs (first rows)")
            st.dataframe(fsq_sample.head(50))

with col2:
    st.header("Options / Notes")
    st.write("- Uses the first parquet file returned by Hugging Face datasets-server for `foursquare/fsq-os-places`.")
    st.write("- DuckDB queries the parquet over HTTPS (httpfs). No full dataset download.")
    st.write("- Geocoding uses Nominatim (rate-limited). For production use, prefer a paid geocoder.")
    st.write("- Census key required: put it into Streamlit secrets as `CENSUS_API_KEY` or set `CENSUS_API_KEY` variable.")

st.markdown("---")

# ---- show the collected data ----
if "locations" in st.session_state and st.session_state.locations:
    df = pd.DataFrame(st.session_state.locations)
    st.subheader("Collected locations")
    st.dataframe(df)

    # regression UI: select predictors
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidate_predictors = [c for c in numeric_cols if c not in ("revenue", "lat", "lon", "radius_m")]
    if not candidate_predictors:
        st.info("No numeric predictors available yet. Add locations (median_income will appear after first add).")
    else:
        st.subheader("Run regression (select predictors)")
        predictors = st.multiselect("Choose predictors (independent variables):", candidate_predictors, default=["median_income"] if "median_income" in candidate_predictors else [candidate_predictors[0]])
        if st.button("Fit regression"):
            model = run_regression(df, predictors)
            if model is None:
                st.warning("Not enough non-null rows to run regression with the selected predictors.")
            else:
                st.subheader("Regression summary")
                st.text(model.summary().as_text())

                # scatter / diagnostics: for single predictor, scatter revenue vs predictor
                if len(predictors) == 1:
                    xcol = predictors[0]
                    df_plot = df.dropna(subset=[xcol, "revenue"])
                    chart = alt.Chart(df_plot).mark_circle(size=80).encode(
                        x=alt.X(xcol, title=xcol),
                        y=alt.Y("revenue", title="Revenue (proxy)"),
                        tooltip=["location", xcol, "revenue"]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

else:
    st.info("No locations yet — add a location to begin.")