from OFL.Predictors.Categories import get_osm_category, get_foursquare_category
from OFL.Predictors import Predictors
from OFL import Helpers
import pandas as pd


def build_inference_features_for_location(lat, lon, radius_m, cr, _fsq_duckdb_con, _fsq_query_cache, census_api_key):
    neighborhood_points = Predictors.generate_circle_points(lat, lon, radius_m, cr)
    print(f'Number of neighborhood points {len(neighborhood_points)}')
    features = []
    for (lat_i, lon_i) in neighborhood_points:
        pop = Helpers.get_population_density_gee(lat_i, lon_i, cr)
        osm_poi = Helpers.get_osm_poi_density(lat_i, lon_i, cr)
        fsq_poi = Predictors.get_fsq_count(lat_i, lon_i, cr, _fsq_query_cache, _fsq_duckdb_con, )
        income = Predictors.get_median_income_by_point(lat_i, lon_i, cr, census_api_key)
        osm_cat = get_osm_category(lat, lon)
        fsq_cat = get_foursquare_category(lat, lon)
        features.append({
            "lat": lat_i,
            "lon": lon_i,
            "population_density": pop,
            "osm_poi_density": osm_poi,
            "fsq_poi_count": fsq_poi,
            "median_income": income,
            "location_category_foursquare": fsq_cat,
            "location_category_osm": osm_cat,
        })

    return pd.DataFrame(features)
