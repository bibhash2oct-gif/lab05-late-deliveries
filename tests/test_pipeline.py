"""Tests for the feature lists, the split and the pipeline."""

import numpy as np
import pytest

from latedeliveries.pipeline import FEATURES, TARGET, split_orders


@pytest.mark.parametrize("column", ["customer_rating", "order_id", TARGET])
def test_feature_list_leaves_out_column(column):
    assert column not in FEATURES


def test_split_is_stratified(split):
    _, _, y_train, y_test = split
    assert len(y_train) == 2250
    assert len(y_test) == 750
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_is_repeatable(orders, split):
    X_train, X_test, _, _ = split
    X_train_again, X_test_again, _, _ = split_orders(orders)
    assert X_train.index.equals(X_train_again.index)
    assert X_test.index.equals(X_test_again.index)


def test_split_keeps_only_feature_columns(split):
    X_train, X_test, _, _ = split
    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES


def test_probabilities_are_valid(fitted_pipeline, split):
    _, X_test, _, _ = split
    probability = fitted_pipeline.predict_proba(X_test)
    assert probability.shape == (len(X_test), 2)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert np.allclose(probability.sum(axis=1), 1)


def test_pipeline_handles_missing_value_and_unseen_category(fitted_pipeline, odd_order):
    probability = fitted_pipeline.predict_proba(odd_order)[0, 1]
    assert 0 <= probability <= 1
