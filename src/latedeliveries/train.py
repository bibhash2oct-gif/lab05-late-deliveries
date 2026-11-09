"""Compare, tune, choose a threshold, test once and save. Run it with:

python -m latedeliveries.train
"""

import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
    threshold_metrics,
)
from latedeliveries.pipeline import (
    CATEGORICAL,
    EXCLUDED,
    NUMERIC,
    SEED,
    build_pipeline,
    split_orders,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SCORING = {
    "accuracy": "accuracy",
    # zero_division=0 stops a warning for the dummy, which never flags anyone
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

# Chosen after reading reports/cv_results.csv. The README explains why.
CHOSEN_MODEL = "logistic"
PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}
TUNING_METRIC = "roc_auc"


def candidates() -> dict[str, BaseEstimator]:
    """The baseline and the real candidates, all unfitted."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000),
        "logistic_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }


def make_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Five-fold CV on the training data only: mean and std of each metric.

    Also records the lowest out-of-fold cost each model can reach, because the
    company cares about dollars, not about any one metric.
    """
    rows = []
    for name, model in candidates().items():
        pipe = build_pipeline(model)
        scores = cross_validate(
            pipe, X_train, y_train, cv=make_folds(), scoring=SCORING
        )
        row: dict[str, object] = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 3)
            row[f"{metric}_std"] = round(float(values.std()), 3)
        probability = cross_val_predict(
            pipe, X_train, y_train, cv=make_folds(), method="predict_proba"
        )[:, 1]
        threshold = choose_threshold(y_train, probability)
        row["best_threshold"] = threshold
        row["cv_cost"] = order_cost(y_train, probability >= threshold)
        rows.append(row)
        print(f"  {name:24s} roc_auc {row['roc_auc_mean']:.3f}  cost ${row['cv_cost']}")
    return pd.DataFrame(rows)


def tune(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Grid search over the chosen model, inside the same pipeline."""
    search = GridSearchCV(
        build_pipeline(candidates()[CHOSEN_MODEL]),
        PARAM_GRID,
        scoring=TUNING_METRIC,
        cv=make_folds(),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1 Split. The test set is not touched again until step 5.
    orders = make_orders()
    X_train, X_test, y_train, y_test = split_orders(orders)
    print(f"Train {len(X_train)} orders, test {len(X_test)} orders")
    print(f"Late rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    # 2 Compare the baseline and the candidates with CV on the training data.
    print("Comparing models with 5-fold CV on the training data")
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS_DIR / "cv_results.csv", index=False)

    # 3 Tune the chosen model.
    search = tune(X_train, y_train)
    untuned = comparison.set_index("model").loc[CHOSEN_MODEL]
    gain = search.best_score_ - untuned[f"{TUNING_METRIC}_mean"]
    beat_spread = bool(gain > untuned[f"{TUNING_METRIC}_std"])
    grid = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    grid.to_csv(REPORTS_DIR / "grid_results.csv", index=False)
    print(f"Best settings {search.best_params_}")
    print(f"Best CV {TUNING_METRIC} {search.best_score_:.3f}")
    print(f"Gain over untuned {gain:+.3f}; beats the fold spread: {beat_spread}")

    # 4 Threshold from out-of-fold probabilities on the training data only.
    tuned = build_pipeline(candidates()[CHOSEN_MODEL]).set_params(**search.best_params_)
    oof_probability = cross_val_predict(
        tuned, X_train, y_train, cv=make_folds(), method="predict_proba"
    )[:, 1]
    cost_by_threshold(y_train, oof_probability).to_csv(
        REPORTS_DIR / "threshold_costs.csv", index=False
    )
    threshold = choose_threshold(y_train, oof_probability)
    print(f"Threshold with the lowest CV cost: {threshold}")

    # 5 Fit on the whole training set and score the test set once.
    tuned.fit(X_train, y_train)
    test_probability = tuned.predict_proba(X_test)[:, 1]
    metrics = threshold_metrics(y_test, test_probability, threshold)
    metrics["costs"] = policy_costs(y_test, test_probability, threshold)
    metrics["test_orders"] = len(y_test)
    metrics["late_orders"] = int(y_test.sum())
    write_json(REPORTS_DIR / "metrics.json", metrics)
    print("Test set, used once:", json.dumps(metrics))

    # 6 Save the whole pipeline and a record of how it was made.
    joblib.dump(tuned, MODELS_DIR / "late_orders.joblib")
    info = {
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "threshold": threshold,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "excluded_columns": EXCLUDED,
        "model": CHOSEN_MODEL,
        "tuning_metric": TUNING_METRIC,
        "best_params": search.best_params_,
        "cv_best_score": round(float(search.best_score_), 3),
        "tuning_beat_fold_spread": beat_spread,
        "test_results": metrics,
    }
    write_json(MODELS_DIR / "model_info.json", info)
    print("Saved models/late_orders.joblib, models/model_info.json and reports/")


if __name__ == "__main__":
    main()
