
def get_nearest_place_coords(lat, lon):
    """
    Returns (lat, lon) of nearest city/town/village center to the input coordinates.
    Uses Nominatim reverse + forward geocoding.
    """
    geolocator = Nominatim(user_agent="geo-fallback-app")

    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location and "address" in location.raw:
            addr = location.raw["address"]
            place = addr.get("city") or addr.get("town") or addr.get("village")
            if place:
                place_loc = geolocator.geocode(place)
                if place_loc:
                    return (place_loc.latitude, place_loc.longitude)
    except Exception as e:
        print(f"Geocoding fallback failed: {e}")

    return None

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
def get_population_density_gee(lat, lon, radius_m, max_expand=3, expand_factor=2):
    """
    Get population density from WorldPop using Earth Engine.
    Expands radius if no values are found, and falls back to nearest
    city/town center if still empty.
    """
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
                return pop_val
        except Exception as e:
            print(f"GEE fallback query failed: {e}")

    print("No population data found, even after fallback.")
    return 0