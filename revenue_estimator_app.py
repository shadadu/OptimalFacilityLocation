# revenue_estimator_app.py
import streamlit as st
import duckdb
import requests
import osmnx as ox
from shapely.geometry import Point
import ee

# ---- Init Earth Engine ----
# ee.Initialize()
ee.Authenticate()
ee.Initialize(project='ee-shaddie77')

# ---- UI Header ----
st.title("📊 Location Revenue Potential Estimator")
st.write("Estimate a simple revenue potential score using open datasets: "
         "Foursquare Open Places (Hugging Face), WorldPop (GEE), and OSM POIs.")

# ---- User Inputs ----
location_name = st.text_input("Enter a location:", "Times Square, New York, NY")
radius_m = st.slider("Search radius (meters)", 100, 2000, 500)

if st.button("Estimate Revenue Potential"):
    # ---- Geocode location ----
    lat, lon = ox.geocoder.geocode(location_name)
    st.map({"lat": [lat], "lon": [lon]})

    # ---- Get Foursquare POI Count from Hugging Face ----
    @st.cache_data
    def query_fsq_count(lat, lon, radius_m):
        api_url = "https://datasets-server.huggingface.co/parquet?dataset=foursquare/fsq-os-places"
        j = requests.get(api_url).json()
        parquet_urls = [f['url'] for f in j.get('parquet_files', []) if f['split'] == 'train']
        if not parquet_urls:
            return 0
        url = parquet_urls[0]

        con = duckdb.connect()
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")

        deg_buffer = radius_m / 111_320  # Approx degrees per meter
        min_lat, max_lat = lat - deg_buffer, lat + deg_buffer
        min_lon, max_lon = lon - deg_buffer, lon + deg_buffer

        query = f"""
        SELECT COUNT(*) as count
        FROM '{url}'
        WHERE latitude BETWEEN {min_lat} AND {max_lat}
          AND longitude BETWEEN {min_lon} AND {max_lon}
        """
        res = con.execute(query).df()
        return int(res['count'][0]) if res.shape[0] else 0

    fsq_poi_count = query_fsq_count(lat, lon, radius_m)

    # ---- Get Population Density from GEE ----
    def get_population_density_gee(lat, lon, radius_m):
        dataset = ee.ImageCollection("WorldPop/GP/100m/pop") \
                     .filter(ee.Filter.date('2020-01-01', '2020-12-31')) \
                     .first()
        point = ee.Geometry.Point(lon, lat)
        region = point.buffer(radius_m).bounds()
        stats = dataset.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=1e9
        )
        
        ans = stats.getInfo().get('population', 0)
        return 0 if ans == None else 0

    population_density_score = get_population_density_gee(lat, lon, radius_m)

    # ---- Get OSM POI Density ----
    def get_osm_poi_density(lat, lon, radius):
        tags = {"amenity": True}
        pois = ox.features_from_point((lat, lon), tags=tags, dist=radius)
        return len(pois)

    poi_density_score = get_osm_poi_density(lat, lon, radius_m)

    # ---- Compute Revenue Score ----
    def estimate_revenue(foot, pop, poi):
        return (foot + 1) * (pop + 1) * (poi + 1)

    estimated_revenue_score = estimate_revenue(fsq_poi_count, population_density_score, poi_density_score)

    # ---- Display Results ----
    st.subheader("📈 Results")
    st.write(f"**Foursquare POI Count (Hugging Face)**: {fsq_poi_count}")
    st.write(f"**Population Density (WorldPop)**: {population_density_score:.2f}")
    st.write(f"**OSM POI Density**: {poi_density_score}")
    st.write(f"**Estimated Revenue Score**: {estimated_revenue_score:.2f}")
