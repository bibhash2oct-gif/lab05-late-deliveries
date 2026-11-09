"""Feature lists, the train/test split and the one shared pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.25
TARGET = "late"

# Only columns whose value is known at the moment the order is placed.
NUMERIC = ["distance_km", "items", "hour", "prep_minutes", "courier_months"]
CATEGORICAL = ["weather", "day_type", "vehicle"]
FEATURES = NUMERIC + CATEGORICAL

# Columns left out on purpose, with the reason. The README explains each one.
EXCLUDED = {
    "order_id": "a label for the order; it says nothing about the delivery",
    "customer_rating": "given after delivery, so unknown when the order is placed",
}


def split_orders(
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 75/25 split. Only the feature columns go into X."""
    X = orders[FEATURES]
    y = orders[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(
    model: BaseEstimator,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    """Preparation and model in one object, so they are fitted and saved together.

    numeric and categorical default to the official feature lists. They exist only
    so the notebook can run the leakage check without editing this file.
    """
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    preprocess = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])
