"""Tests for the cost rule and the threshold choice, on examples worked on paper."""

import pytest

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
)

# Five orders. Two are late (the first and the third).
Y_TRUE = [1, 0, 1, 0, 0]
PROBABILITY = [0.9, 0.6, 0.4, 0.2, 0.1]


def test_cost_of_a_small_example():
    y_true = [1, 1, 0, 0, 1, 0]
    flagged = [True, False, True, False, False, False]
    # two vouchers (2 x 4) and two late orders missed (2 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 2 * 15


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.10, 5 * 4),  # all five flagged, nothing missed
        (0.20, 4 * 4),  # four flagged, nothing missed
        (0.30, 3 * 4),  # three flagged, nothing missed
        (0.50, 2 * 4 + 15),  # two flagged, the 0.4 order is missed
        (0.70, 1 * 4 + 15),  # one flagged, the 0.4 order is missed
        (0.95, 2 * 15),  # nobody flagged, both late orders missed
    ],
)
def test_cost_at_each_threshold(threshold, expected):
    table = cost_by_threshold(Y_TRUE, PROBABILITY, thresholds=[threshold])
    assert table.loc[0, "cost"] == expected


def test_threshold_choice_on_a_small_example():
    # 0.25 to 0.40 all cost $12, the lowest; ties go to the lowest threshold
    assert choose_threshold(Y_TRUE, PROBABILITY) == 0.25


def test_threshold_can_choose_to_flag_nobody():
    # no order is late, so every voucher is wasted
    y_true = [0, 0, 0, 0]
    probability = [0.5, 0.6, 0.7, 0.8]
    threshold = choose_threshold(y_true, probability)
    assert all(p < threshold for p in probability)
    assert order_cost(y_true, [p >= threshold for p in probability]) == 0


def test_threshold_can_choose_to_flag_everybody():
    # every order is late, so missing any one costs $15 instead of $4
    y_true = [1, 1, 1, 1]
    probability = [0.05, 0.2, 0.3, 0.4]
    threshold = choose_threshold(y_true, probability)
    assert threshold == 0.05
    assert all(p >= threshold for p in probability)


def test_policy_costs():
    costs = policy_costs(Y_TRUE, PROBABILITY, threshold=0.3)
    assert costs == {"flag_nobody": 30, "flag_everybody": 20, "model": 12}
