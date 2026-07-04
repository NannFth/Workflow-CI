import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor


def load_data():
    train_data = pd.read_csv("california_housing_preprocessing/train.csv")
    test_data = pd.read_csv("california_housing_preprocessing/test.csv")

    X_train = train_data.drop("median_house_value", axis=1)
    y_train = train_data["median_house_value"]

    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"]

    return X_train, X_test, y_train, y_test


def main():
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("R2 Score pada data test:", score)


if __name__ == "__main__":
    main()