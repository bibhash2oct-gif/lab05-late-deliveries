"""Metrics, the cost rule and the threshold choice."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER_COST = 4  # paid for every order we flag
LATE_COST = 15  # paid for every late order we did not flag
THRESHOLDS = np.round(np.arange(1, 20) * 0.05, 2)  # 0.05, 0.10, ... 0.95


def order_cost(y_true, flagged) -> int:
    """$4 for every flagged order plus $15 for every late order not flagged."""
    late = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    vouchers = int(flagged.sum())
    missed = int((late & ~flagged).sum())
    return VOUCHER_COST * vouchers + LATE_COST * missed


def cost_by_threshold(y_true, probability, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Total cost of flagging every order whose probability is >= each threshold."""
    probability = np.asarray(probability)
    rows = [
        {
            "threshold": float(t),
            "flagged": int((probability >= t).sum()),
            "cost": order_cost(y_true, probability >= t),
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def choose_threshold(y_true, probability, thresholds=THRESHOLDS) -> float:
    """The threshold with the lowest total cost. On a tie, the lowest threshold."""
    table = cost_by_threshold(y_true, probability, thresholds)
    best_row = table["cost"].idxmin()  # idxmin returns the first lowest row
    return float(table.loc[best_row, "threshold"])


def policy_costs(y_true, probability, threshold: float) -> dict[str, int]:
    """Cost of flagging nobody, flagging everybody and following the model."""
    n = len(y_true)
    return {
        "flag_nobody": order_cost(y_true, np.zeros(n, dtype=bool)),
        "flag_everybody": order_cost(y_true, np.ones(n, dtype=bool)),
        "model": order_cost(y_true, np.asarray(probability) >= threshold),
    }


def threshold_metrics(y_true, probability, threshold: float) -> dict:
    """Confusion matrix, precision, recall, F1 and ROC-AUC at a chosen threshold."""
    flagged = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "precision": round(float(precision_score(y_true, flagged, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, flagged)), 3),
        "f1": round(float(f1_score(y_true, flagged)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 3),
    }
