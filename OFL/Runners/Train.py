from OFL.Predictors.Predictors import build_features_for_location
from OFL.Predictors.Categories import encode_location_categories
from OFL.RevenueEstimation.RevenueEstimation import revenue_estimation
def build_train_vars(candidates, radius_m, cr):
    rows = []
    cnt_ = 0
    for lat, lon in candidates:
        # print(f'cnt_ {cnt_}')
        X_df = build_features_for_location(lat, lon, radius_m, cr)
        # Aggregate neighborhood features (mean as example)
        agg = X_df.mean(numeric_only=True).to_dict()
        Y = revenue_estimation
        print(f'revenue {Y}')
        agg["lat"], agg["lon"], agg["revenue"] = lat, lon, Y,
        # agg["lat"], agg["lon"] = lat, lon
        rows.append(agg)
        cnt_ += 1

    return rows

def build_xy(rows):
    df = pd.DataFrame(rows)
    df_vars = encode_location_categories(df)
    X = df_vars[["population_density"
        , "osm_poi_density"
        , "fsq_poi_count"
        , "median_income"
        , "fsq_category_encoded"
        , "osm_category_encoded"]]
    y = df_vars["revenue"]

    return X, y

def train(model, X, y):
    if model is None:
        model = LinearRegression().fit(X, y)
        print("Regression coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        return model
    else:
        return None
