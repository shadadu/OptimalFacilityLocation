import duckdb
import osmnx as ox
import ee
import requests
from geopy.geocoders import Nominatim
import time

# Initialize Earth Engine
# ee.Authenticate()
# ee.Initialize(project='ee-shaddie77')

# # In-memory caches
_geocode_cache = {}
_pop_cache = {}



# Helper: robust cached geocoding
def get_nearest_place_coords(lat, lon):
    """
    Returns (lat, lon) of nearest city/town/village center.
    Uses Nominatim reverse + forward geocoding, with caching + backoff.
    """
    key = f"{lat:.5f}_{lon:.5f}"
    if key in _geocode_cache:
        return _geocode_cache[key]

    geolocator = Nominatim(user_agent="geo-fallback-app", timeout=10)

    try:
        # Reverse lookup (lat, lon -> nearest place)
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and "address" in location.raw:
            addr = location.raw["address"]
            place = addr.get("city") or addr.get("town") or addr.get("village")
            if place:
                # Forward geocode the place name to get its centroid
                place_loc = geolocator.geocode(place)
                if place_loc:
                    coords = (place_loc.latitude, place_loc.longitude)
                    _geocode_cache[key] = coords
                    return coords
    except Exception as e:
        print(f"Geocoding fallback failed: {e}")
        time.sleep(2)  # backoff

    _geocode_cache[key] = None
    return None


def get_population_density_gee(lat, lon, radius_m, max_expand=3, expand_factor=2):
    """
    Get population density from WorldPop using Earth Engine.
    Expands radius if no values are found, and falls back to nearest
    city/town center if still empty.
    Includes caching to avoid repeated queries.
    """
    cache_key = f"{lat:.5f}_{lon:.5f}_{radius_m}"
    if cache_key in _pop_cache:
        return _pop_cache[cache_key]

    print(f"Getting population density at ({lat}, {lon}), radius={radius_m}m")

    dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filter(ee.Filter.date('2020-01-01', '2020-12-31')) \
        .first()

    attempt_radius = radius_m

    # Try with expanding radius
    for attempt in range(max_expand + 1):
        try:
            point = ee.Geometry.Point(lon, lat)
            region = point.buffer(attempt_radius).bounds()
            stats = dataset.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            )
            result = stats.getInfo()
            pop_val = result.get('population', None)
            if pop_val is not None:
                print(f'Found pop_val {pop_val}')
                _pop_cache[cache_key] = pop_val
                return pop_val
            else:
                print(f"No population data at radius {attempt_radius}m, expanding search...")
        except Exception as e:
            print(f"GEE query failed at radius {attempt_radius}m: {e}")

        attempt_radius *= expand_factor

    # Fallback to nearest city/town
    print("No population found after expansions. Falling back to nearest town/city center...")
    fallback_coords = get_nearest_place_coords(lat, lon)
    if fallback_coords:
        try:
            point = ee.Geometry.Point(fallback_coords[::-1])  # (lon, lat)
            region = point.buffer(radius_m).bounds()
            stats = dataset.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            )
            result = stats.getInfo()
            pop_val = result.get('population', None)
            if pop_val is not None:
                print(f'Found pop_val {pop_val}')
                _pop_cache[cache_key] = pop_val
                return pop_val
        except Exception as e:
            print(f"GEE fallback query failed: {e}")

    print("No population data found, even after fallback.")
    _pop_cache[cache_key] = 0
    return 0

# ----------------------------
# POI Density (with fallback)
# ----------------------------
def get_osm_poi_density(lat, lon, radius, max_expand=3, expand_factor=2):
    """
    Get POI density from OSM.
    Expands radius if no POIs found, and falls back to nearest
    town/city center if still empty.
    """
    print(f"Getting POI density at ({lat}, {lon}), radius={radius}m")

    attempt_radius = radius
    tags = {"amenity": True}

    # Try with expanding radius
    for attempt in range(max_expand + 1):
        try:
            pois = ox.features_from_point((lat, lon), tags=tags, dist=attempt_radius)
            if len(pois) > 0:
                print(f'Found osm_pois {len(pois)}')
                return len(pois)
            else:
                print(f"No POIs at radius {attempt_radius}m, expanding search...")
        except Exception as e:
            print(f"OSM query failed at radius {attempt_radius}m: {e}")

        attempt_radius *= expand_factor

    # Fallback to nearest city/town
    print("No POIs found after expansions. Falling back to nearest town/city center...")
    fallback_coords = get_nearest_place_coords(lat, lon)
    if fallback_coords:
        try:
            pois = ox.features_from_point(fallback_coords, tags=tags, dist=radius)
            if len(pois) > 0:
                print(f'Found osm poi {len(pois)}')
                return len(pois)
        except Exception as e:
            print(f"OSM fallback query failed: {e}")

    print("No POIs found, even after fallback.")
    return 0

def snap_to_nearest_town(lat, lon):
    """
    Snap (lat, lon) to nearest town/city center if available.
    """
    geolocator = Nominatim(user_agent="geo_fallback")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location and "town" in location.raw["address"]:
            town = location.raw["address"]["town"]
        elif location and "city" in location.raw["address"]:
            town = location.raw["address"]["city"]
        else:
            return lat, lon  # no town/city info, return same coords

        # Forward geocode the town name → town center coords
        town_loc = geolocator.geocode(town)
        if town_loc:
            return town_loc.latitude, town_loc.longitude
    except Exception as e:
        print(f"Town fallback failed: {e}")
    return lat, lon  # fallback to original point

def get_fips_from_coords(lat, lon, retries=3, wait=5):
    """
        Try FCC API first. If it fails, fallback to Census Geocoder API.
        Returns block info JSON.
        """
    fcc_url = "https://geo.fcc.gov/api/census/block/find"
    params = {"latitude": lat, "longitude": lon, "format": "json"}

    # Try FCC first with retry logic
    for attempt in range(retries):
        try:
            resp = requests.get(fcc_url, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            # if last attempt, fall back
            if attempt == retries - 1:
                break
            time.sleep(2 ** attempt)  # exponential backoff

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
    try:
        resp = requests.get(geocoder_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Extract equivalent info
        block = data["result"]["geographies"]["Census Blocks"][0]
        return {
            "Block": {"FIPS": block["GEOID"]},
            "County": {"FIPS": block["COUNTY"]},
            "State": {"FIPS": block["STATE"]},
            "Source": "Census Geocoder"
        }
    except Exception as e:
        raise RuntimeError("Both FCC and Census Geocoder failed") from e


def _get_duckdb_connection(_fsq_duckdb_con):
    """Create and cache a DuckDB connection."""
    # global _fsq_duckdb_con
    if _fsq_duckdb_con is None:
        con = duckdb.connect()
        _fsq_duckdb_con = con
    return _fsq_duckdb_con