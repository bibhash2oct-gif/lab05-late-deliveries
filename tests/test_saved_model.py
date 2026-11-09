"""A saved pipeline must predict exactly as it did before saving."""

import joblib
import numpy as np


def test_saved_model_predicts_the_same_after_reloading(
    fitted_pipeline, split, tmp_path
):
    _, X_test, _, _ = split
    path = tmp_path / "late_orders.joblib"
    joblib.dump(fitted_pipeline, path)
    loaded = joblib.load(path)
    assert np.array_equal(
        loaded.predict_proba(X_test), fitted_pipeline.predict_proba(X_test)
    )


def test_saved_model_handles_an_odd_order(fitted_pipeline, odd_order, tmp_path):
    path = tmp_path / "late_orders.joblib"
    joblib.dump(fitted_pipeline, path)
    loaded = joblib.load(path)
    assert np.array_equal(
        loaded.predict_proba(odd_order), fitted_pipeline.predict_proba(odd_order)
    )
