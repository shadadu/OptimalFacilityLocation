import requests
import math
from geopy.distance import geodesic
from functools import lru_cache

NYC_OD_BASE = "https://data.cityofnewyork.us/resource"
REQ_TIMEOUT = 30
HEADERS = {"Accept": "application/json"}

# --- Simple cache dict ---
_geoclient_cache = {}


def geoclient_reverse_geocode(lat, lon, app_id: str, app_key: str):
    """
    Reverse geocode lat/lon to BBL using NYC Geoclient API.
    Uses in-memory cache to avoid duplicate API calls.
    """
    # Round to ~6 decimal places for stable cache key
    key = (round(lat, 6), round(lon, 6))
    if key in _geoclient_cache:
        return _geoclient_cache[key]

    url = f"https://api.nyc.gov/geo/geoclient/v1/latlon.json"
    params = {
        "lat": lat,
        "lon": lon,
        "app_id": app_id,
        "app_key": app_key
    }
    r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "latlon" in data and "bbl" in data["latlon"]:
        bbl_data = data["latlon"]
        bbl = (int(bbl_data["boroughCode"]), int(bbl_data["block"]), int(bbl_data["lot"]))
        _geoclient_cache[key] = bbl
        return bbl

    _geoclient_cache[key] = None
    return None


def nyc_dof_assessment_by_bbl(boro: int, block: int, lot: int, app_token: str | None = None):
    """
    Query NYC DOF Property Valuation & Assessment dataset by BBL.
    """
    url = f"{NYC_OD_BASE}/yjxr-fw8i.json"
    params = {"boro": str(boro), "block": str(block), "lot": str(lot)}
    headers = HEADERS.copy()
    if app_token:
        headers["X-App-Token"] = app_token
    r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()


def revenue_estimation_by_dof_assessment(lat, lon
                                         , geoclient_app_id
                                         , geoclient_app_key
                                         , socrata_app_token=None,
                                         radius_tax_value=200):
    """
    Estimate revenue using NYC DOF assessment values.
    1. Convert lat/lon to BBL (cached).
    2. Query DOF dataset for assessment value.
    3. If not available, search within radius_tax_value for nearest property with data.
    """
    # Step 1: Try direct match via Geoclient (cached)
    try:
        bbl = geoclient_reverse_geocode(lat, lon, geoclient_app_id, geoclient_app_key)
        if bbl:
            boro, block, lot = bbl
            result = nyc_dof_assessment_by_bbl(boro, block, lot, app_token=socrata_app_token)
            if result and "marketvalue" in result[0]:
                return float(result[0]["marketvalue"])
            elif result and "assessedvalue" in result[0]:
                return float(result[0]["assessedvalue"])
    except Exception as e:
        print(f"Direct BBL lookup failed: {e}")

    # Step 2: Nearby search fallback
    try:
        url = f"{NYC_OD_BASE}/yjxr-fw8i.json"
        lat_min = lat - (radius_tax_value / 111320)  # degrees latitude
        lat_max = lat + (radius_tax_value / 111320)
        lon_min = lon - (radius_tax_value / (40075000 * math.cos(math.radians(lat)) / 360))
        lon_max = lon + (radius_tax_value / (40075000 * math.cos(math.radians(lat)) / 360))

        params = {
            "$where": f"latitude between {lat_min} and {lat_max} AND longitude between {lon_min} and {lon_max}",
            "$limit": 50
        }
        headers = HEADERS.copy()
        if socrata_app_token:
            headers["X-App-Token"] = socrata_app_token

        r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if data:
            nearest = min(
                data,
                key=lambda d: geodesic(
                    (lat, lon),
                    (float(d["latitude"]), float(d["longitude"]))
                ).meters
            )
            if "marketvalue" in nearest:
                return float(nearest["marketvalue"])
            elif "assessedvalue" in nearest:
                return float(nearest["assessedvalue"])
    except Exception as e:
        print(f"Nearby search failed: {e}")

    return float("nan")
