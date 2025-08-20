from sklearn.preprocessing import LabelEncoder
import pandas as pd

# shared encoders so model always uses the same mapping
_fsq_encoder = LabelEncoder()
_osm_encoder = LabelEncoder()
_encoders_fitted = False

def encode_location_categories(df):
    """
    Encode Foursquare + OSM category labels into numeric values
    so they can be used as regression features.

    Expects df with columns:
        - location_category_foursquare
        - location_category_osm
    Returns the same df with numeric-encoded columns added.
    """
    global _encoders_fitted

    # Ensure columns exist, replace None with "unknown"
    for col in ["location_category_foursquare", "location_category_osm"]:
        if col not in df:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown")

    # Fit encoders once on all available categories
    if not _encoders_fitted:
        _fsq_encoder.fit(df["location_category_foursquare"].unique())
        _osm_encoder.fit(df["location_category_osm"].unique())
        _encoders_fitted = True

    # Transform into numeric codes
    df["fsq_category_encoded"] = _fsq_encoder.transform(df["location_category_foursquare"])
    df["osm_category_encoded"] = _osm_encoder.transform(df["location_category_osm"])

    return df

def category_with_fallback(lat, lon, fetch_fn, radii=[200, 500, 1000, 2000], delay=1):
    """
    Try to fetch category with expanding radius.
    If still empty, snap to nearest town and retry once.

    fetch_fn: function(lat, lon, radius) -> str | None
    """
    for r in radii:
        try:
            category = fetch_fn(lat, lon, r)
            if category:  # got something
                return category
        except Exception as e:
            print(f"Fetch attempt failed at radius {r}: {e}")
        time.sleep(delay)

    # Snap to nearest town & retry once
    town_lat, town_lon = snap_to_nearest_town(lat, lon)
    if (town_lat, town_lon) != (lat, lon):
        return category_with_fallback(town_lat, town_lon, fetch_fn, radii, delay)

    return "Unknown"
def snap_to_nearest_town(lat, lon):
    """
    Snap (lat, lon) to nearest town/city center if available.
    """
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



def _fetch_foursquare_category(lat, lon, radius, max_radius=5000):
    """
    Fetch category from Foursquare (Hugging Face parquet + DuckDB).
    Expands radius if no result found, and falls back to nearest town center if still none.
    """
    try:
        # Load parquet metadata
        api_url = "https://datasets-server.huggingface.co/parquet?dataset=foursquare/fsq-os-places"
        j = requests.get(api_url).json()
        parquet_urls = [f['url'] for f in j.get('parquet_files', []) if f['split'] == 'train']

        if not parquet_urls:
            return None

        con = duckdb.connect()
        # Install spatial extension if not already
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        # Try radius expansion
        search_radius = radius
        while search_radius <= max_radius:
            query = f"""
            SELECT fsq_category_labels[1], latitude, longitude
            FROM parquet_scan({parquet_urls})
            WHERE ST_DWithin(
                ST_Point(longitude, latitude),
                ST_Point({lon}, {lat}),
                {search_radius}
            )
            LIMIT 1;
            """
            try:
                res = con.execute(query).fetchone()
            except Exception as e:
                print(f"Error querying DuckDB: {e}")
                return None

            if res:
                category, clat, clon = res
                return category

            search_radius *= 2  # expand radius

        # 🔹 Fallback: snap to nearest town and retry once
        town_lat, town_lon = snap_to_nearest_town(lat, lon)
        if (town_lat, town_lon) != (lat, lon):
            return _fetch_foursquare_category(town_lat, town_lon, radius, max_radius)

        return None

    except Exception as e:
        print(f"Error fetching Foursquare category: {e}")
        return None

def get_foursquare_category(lat, lon):
    return category_with_fallback(lat, lon, _fetch_foursquare_category)

def _fetch_osm_category(lat, lon, radius):
    tags = {"amenity": True, "shop": True, "landuse": True}
    pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
    if len(pois) > 0:
        for key in ["amenity", "shop", "landuse"]:
            if key in pois.columns:
                values = pois[key].dropna().unique()
                if len(values) > 0:
                    return values[0]
    return None

def get_osm_category(lat, lon):
    return category_with_fallback(lat, lon, _fetch_osm_category)