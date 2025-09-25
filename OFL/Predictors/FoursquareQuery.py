import os
import duckdb
import hashlib

# Globals / caches
# _fsq_query_cache = {}
# _fsq_duckdb_con = None
_fsq_local_file = None
_coord_columns_cache = {}  # cache which pair of coordinate cols works


def _ensure_local_parquet(local_path="//Users//rckyi//Documents//Data//fsq_places.parquet"):
    global _fsq_local_file
    if _fsq_local_file is not None:
        return _fsq_local_file
    if not os.path.exists(local_path):
        # download as before; ensure we have local file
        # (you already have a function like this — integrate it)
        raise FileNotFoundError(f"Local FSQ parquet not found at {local_path}")
    _fsq_local_file = local_path
    return _fsq_local_file


# def _get_duckdb_con():
#     global _fsq_duckdb_con
#     if _fsq_duckdb_con is None:
#         con = duckdb.connect()
#         _fsq_duckdb_con = con
#     return _fsq_duckdb_con


import duckdb
import hashlib
import os

def _detect_lat_lon_columns(local_file, con):
    """
    Inspect the parquet schema and return the best matching lat/lon columns.
    """
    schema_df = con.execute(f"SELECT * FROM read_parquet('{local_file}') LIMIT 0").fetchdf()
    cols = schema_df.columns.str.lower().tolist()

    lat_candidates = [c for c in cols if "lat" in c]
    lon_candidates = [c for c in cols if "lon" in c or "lng" in c]

    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]

    # If geometry column exists instead
    if "geom" in cols or "geometry" in cols:
        raise ValueError("Parquet has geometry but no lat/lon — need to parse WKT/WKB")

    raise ValueError(f"Could not detect latitude/longitude columns. Found: {cols}")


def get_fsq_count(lat, lon, r, _fsq_query_cache, _fsq_duckdb_con
                  , local_file="/Users/rckyi/Documents/Data/fsq_places.parquet"):
    """
    Count FSQ places within radius r (meters) of lat/lon.
    Uses local parquet + cache to avoid repeated requests.
    """
    # Cache key
    key = hashlib.md5(f"{lat:.6f}_{lon:.6f}_{r}".encode()).hexdigest()
    if key in _fsq_query_cache:
        print(f'_fsq_query_cache[key] {_fsq_query_cache[key]}')
        return _fsq_query_cache[key]

    print("Getting Foursquare Count")

    # Auto-detect lat/lon columns
    lat_col, lon_col = _detect_lat_lon_columns(local_file, _fsq_duckdb_con)
    print(f"✅ Using columns: {lat_col}, {lon_col}")

    # Convert radius meters to degrees (approx)
    deg = r / 111_320
    min_lat, max_lat = lat - deg, lat + deg
    min_lon, max_lon = lon - deg, lon + deg

    query = f"""
    SELECT COUNT(*) as count
    FROM read_parquet('{local_file}')
    WHERE {lat_col} BETWEEN {min_lat} AND {max_lat}
      AND {lon_col} BETWEEN {min_lon} AND {max_lon}
    """

    res = _fsq_duckdb_con.execute(query).fetchdf()
    count = int(res['count'][0]) if res.shape[0] else 0

    _fsq_query_cache[key] = count
    print(f'fsq count {count}')
    return count

