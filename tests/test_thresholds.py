import numpy as np

from fraud_detection.evaluation import select_min_cost_threshold, threshold_cost_curve


def test_selects_threshold_with_lowest_business_cost():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    curve = threshold_cost_curve(
        y_true,
        y_proba,
        false_positive_cost=1,
        false_negative_cost=10,
        thresholds=[0.2, 0.5, 0.8],
    )

    best = select_min_cost_threshold(curve)

    assert best["threshold"] == 0.5
    assert best["total_cost"] == 0

