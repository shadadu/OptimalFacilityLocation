import pandas as pd
from sklearn.linear_model import LinearRegression
from OFL.Predictors.Categories import encode_location_categories
import time
from huggingface_hub import notebook_login
import json, ast
import pickle


def build_xy(data_dir_path):
    file = "location_revenue_and_predictors.csv"
    df = pd.read_csv(data_dir_path + file)
    df_vars = encode_location_categories(df)
    X = df_vars[["population_density"
        , "osm_poi_density"
        , "fsq_poi_count"
        , "median_income"
        , "fsq_category_encoded"
        , "osm_category_encoded"]]
    y = df_vars["revenue"].apply(float)
    print(f'y {y}')

    return X, y


def train(X, y, model):
    if model is None:
        model = LinearRegression().fit(X, y)
        print("Regression coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        return model
    else:
        return None


def main():
    """
    Loads the model from csv file and retrieves X and y for model training
    """
    X, y = build_xy(data_dir_path="/Users/rckyi/Documents/Data/")

    model = train(X, y, None)

    filename = "/Users/rckyi/Documents/Data/linear_regression_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)




if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    print(f"Execution time: {elapsed_minutes:.2f} minutes")
