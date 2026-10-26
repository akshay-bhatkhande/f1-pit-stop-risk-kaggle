"""Metric helper bounds on toy labels/scores."""

import numpy as np

from src.eval.metrics import classification_report_scores, recall_at_topk


def test_metrics_in_unit_interval():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    # Ensure both classes
    y[0], y[1] = 0, 1
    p = rng.random(200)
    scores = classification_report_scores(y, p)
    for k, v in scores.items():
        assert 0.0 <= v <= 1.0, f"{k}={v}"


def test_recall_at_topk_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.0])
    # top 20% = 2 rows → both positives
    assert recall_at_topk(y, p, 0.2) == 1.0
