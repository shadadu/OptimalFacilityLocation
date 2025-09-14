import requests
import certifi
import hashlib
import json
import os
import time
from datetime import datetime, timedelta

# Config
_GEOCODE_CACHE_FILE = "/Users/rckyi/Documents/Data/geocode_cache.json"
_GEOCODE_CACHE_EXPIRY_DAYS = 30   # configurable expiry in days

# In-memory cache (normalized)
_geocode_cache = {}

def _load_cache():
    """Load and normalize on-disk cache into _geocode_cache."""
    global _geocode_cache
    _geocode_cache = {}
    if not os.path.exists(_GEOCODE_CACHE_FILE):
        return

    try:
        with open(_GEOCODE_CACHE_FILE, "r") as f:
            raw = json.load(f)
    except Exception:
        return

    now_iso = datetime.utcnow().isoformat()
    for k, v in raw.items():
        # v can be several forms (legacy list [lat,lon] or new dict)
        if isinstance(v, dict):
            # Expect keys "latlon" and "timestamp"
            latlon = v.get("latlon")
            ts = v.get("timestamp")
            if isinstance(latlon, (list, tuple)) and ts:
                try:
                    # validate timestamp parseable
                    datetime.fromisoformat(ts)
                    _geocode_cache[k] = {"latlon": (float(latlon[0]), float(latlon[1])), "timestamp": ts}
                except Exception:
                    # if timestamp invalid, treat as fresh with now
                    _geocode_cache[k] = {"latlon": (float(latlon[0]), float(latlon[1])), "timestamp": now_iso}
            else:
                # skip malformed dict entries
                continue
        elif isinstance(v, list) and len(v) == 2:
            # legacy: [lat, lon] -> convert to dict format with current timestamp
            try:
                _geocode_cache[k] = {"latlon": (float(v[0]), float(v[1])), "timestamp": now_iso}
            except Exception:
                continue
        else:
            # unknown format -> skip
            continue

def _save_cache():
    """Persist normalized cache to disk (convert latlon tuples to lists for JSON)."""
    serializable = {}
    for k, v in _geocode_cache.items():
        serializable[k] = {"latlon": [v["latlon"][0], v["latlon"][1]], "timestamp": v["timestamp"]}
    tmp = _GEOCODE_CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(serializable, f)
    os.replace(tmp, _GEOCODE_CACHE_FILE)

# load/normalize cache at import/run time
_load_cache()

def geocode_direct(geolocation_name, use_cache=True, rate_limit=1.0):
    """
    Geocode a place name into (lat, lon) using Nominatim directly (requests + certifi).
    Caches results on disk with expiry to avoid repeated hits.
    Respects a simple rate limit (default: 1s) before making external requests.
    """
    key = hashlib.sha1(geolocation_name.strip().lower().encode()).hexdigest()
    now = datetime.utcnow()

    # Cache check with expiry (normalized entries)
    if use_cache and key in _geocode_cache:
        entry = _geocode_cache[key]

        # entry should be dict {"latlon": (lat, lon), "timestamp": iso}
        if isinstance(entry, dict) and "latlon" in entry and "timestamp" in entry:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if now - ts < timedelta(days=_GEOCODE_CACHE_EXPIRY_DAYS):
                    return tuple(entry["latlon"])
                else:
                    # expired -> drop and continue to requery
                    del _geocode_cache[key]
            except Exception:
                # malformed timestamp -> drop and requery
                del _geocode_cache[key]
        else:
            # fallback: if unexpectedly not dict, remove and requery
            try:
                del _geocode_cache[key]
            except Exception:
                pass

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": geolocation_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "revenue_estimator_app"}

    try:
        # polite rate limiting before making request
        if rate_limit and rate_limit > 0:
            time.sleep(rate_limit)

        resp = requests.get(url, params=params, headers=headers, verify=certifi.where(), timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise ValueError(f"Could not geocode location: {geolocation_name}")

        latlon = (float(results[0]["lat"]), float(results[0]["lon"]))

        # Cache with timestamp
        if use_cache:
            _geocode_cache[key] = {"latlon": latlon, "timestamp": now.isoformat()}
            try:
                _save_cache()
            except Exception:
                # non-fatal: caching failure shouldn't crash geocoding
                pass

        return latlon

    except Exception as e:
        # provide a clear error message
        raise RuntimeError(f"Geocoding failed for '{geolocation_name}': {e}")
