"""ROC-AUC, PR-AUC, recall@top-k."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_topk(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    """Fraction of positives captured in the top floor(k_frac * n) scored rows."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(np.floor(k_frac * n)))
    order = np.argsort(-y_score)[:k]
    selected = y_true[order]
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    return float(selected.sum() / n_pos)


def classification_report_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_fracs: Iterable[float] = (0.05, 0.10),
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out: dict[str, float] = {}
    # Guard single-class folds
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    for kf in k_fracs:
        key = f"recall@top{int(kf * 100)}%"
        out[key] = recall_at_topk(y_true, y_score, kf)
    return out


def mean_std_table(fold_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = fold_dicts[0].keys()
    table: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([d[k] for d in fold_dicts], dtype=float)
        table[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=0)),
        }
    return table
