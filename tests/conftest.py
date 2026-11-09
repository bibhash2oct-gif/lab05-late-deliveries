"""Shared fixtures for the tests."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline, split_orders


@pytest.fixture(scope="session")
def orders() -> pd.DataFrame:
    return make_orders()


@pytest.fixture(scope="session")
def split(orders):
    return split_orders(orders)


@pytest.fixture
def fitted_pipeline(split) -> Pipeline:
    X_train, _, y_train, _ = split
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)


@pytest.fixture
def odd_order() -> pd.DataFrame:
    """One order with a missing prep time and a weather value never seen."""
    return pd.DataFrame(
        [
            {
                "distance_km": 4.2,
                "items": 3,
                "hour": 19,
                "prep_minutes": float("nan"),
                "courier_months": 12,
                "weather": "hail",
                "day_type": "weekend",
                "vehicle": "bike",
            }
        ]
    )
