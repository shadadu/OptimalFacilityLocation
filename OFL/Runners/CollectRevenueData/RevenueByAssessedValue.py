import requests
import time
from functools import lru_cache

# --- Global caches (dict-based for flexibility) ---
_census_cache = {}
_revenue_cache = {}


def get_census_block(lat, lon, retries=3):
    """
    Get census block info with FCC -> Census Geocoder fallback.
    Cached by (lat, lon).
    """
    key = (round(lat, 6), round(lon, 6))  # reduce floating point noise
    if key in _census_cache:
        return _census_cache[key]

    fcc_url = "https://geo.fcc.gov/api/census/block/find"
    params = {"latitude": lat, "longitude": lon, "format": "json"}

    # Try FCC first
    for attempt in range(retries):
        try:
            resp = requests.get(fcc_url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            _census_cache[key] = data
            return data
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                break
            time.sleep(2 ** attempt)

    # ---- FALLBACK: Census Geocoder ----
    print("⚠️ FCC failed, falling back to Census Geocoder...")

    geocoder_url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": lon,
        "y": lat,
        "benchmark": "Public_AR_Census2020",
        "vintage": "Census2020_Census2020",
        "format": "json"
    }
    resp = requests.get(geocoder_url, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    block = data["result"]["geographies"]["Census Blocks"][0]
    normalized = {
        "Block": {"FIPS": block["GEOID"]},
        "County": {"FIPS": block["COUNTY"]},
        "State": {"FIPS": block["STATE"]},
        "Source": "Census Geocoder"
    }

    _census_cache[key] = normalized
    return normalized


def revenue_estimation_by_dof_assessment(lat, lon, socrata_app_token):
    """
    Example pipeline: uses census block, then queries Socrata DOF dataset.
    Fully cached by (lat, lon).
    """

    key = (round(lat, 6), round(lon, 6))
    if key in _revenue_cache:
        return _revenue_cache[key]

    block_info = get_census_block(lat, lon)
    fips = block_info["Block"]["FIPS"]
    print(f"✅ Got Census Block FIPS {fips} (source={block_info.get('Source','FCC')})")

    # Example query to Socrata dataset
    url = "https://data.cityofnewyork.us/resource/yjxr-fw8i.json"
    headers = {"X-App-Token": socrata_app_token}
    params = {"$limit": 50, "bbl": fips[:10]}  # adapt schema as needed

    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    print(f'Revenue by dof response {resp}')
    results = resp.json()

    _revenue_cache[key] = results
    return results