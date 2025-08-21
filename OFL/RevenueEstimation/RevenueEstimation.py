from OFL.Helpers import get_osm_poi_density, get_population_density_gee, get_fsq_count

def revenue_estimation(lat, lon):
    print(f'Revenue estimation ...')
    revenue_pop_density_gee = get_population_density_gee(lat, lon, 500)
    revenue_osm_poi = get_osm_poi_density(lat, lon, 500)
    revenue_fsq_count = get_fsq_count(lat, lon, 500)
    return (revenue_pop_density_gee * 2 +
            revenue_osm_poi * 100 +
            revenue_fsq_count * 50
            )