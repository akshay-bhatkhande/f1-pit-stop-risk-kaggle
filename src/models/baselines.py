"""B0 prevalence and B1 simple rule baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PrevalenceBaseline:
    """Predict constant prior P = mean(y) on train fold."""

    def __init__(self) -> None:
        self.prior_: float = 0.2

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> PrevalenceBaseline:
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        p = np.full(n, self.prior_, dtype=float)
        return np.column_stack([1 - p, p])


class RuleBaseline:
    """
    B1: flag high tyre-life quantile and/or late race progress.
    Score is a heuristic in [0, 1] for ranking (not calibrated).
    """

    def __init__(self, tyre_q: float = 0.75, late_progress: float = 0.65) -> None:
        self.tyre_q = tyre_q
        self.late_progress = late_progress
        self.tyre_threshold_: float = 15.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RuleBaseline:
        if "TyreLife" in X.columns:
            self.tyre_threshold_ = float(np.nanquantile(X["TyreLife"], self.tyre_q))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(X), dtype=float)
        if "TyreLife" in X.columns:
            score += 0.45 * (X["TyreLife"].to_numpy(dtype=float) >= self.tyre_threshold_)
        if "RaceProgress" in X.columns:
            score += 0.35 * (
                X["RaceProgress"].to_numpy(dtype=float) >= self.late_progress
            )
        if "late_race" in X.columns:
            score += 0.15 * X["late_race"].to_numpy(dtype=float)
        if "track_caution" in X.columns:
            score += 0.20 * X["track_caution"].to_numpy(dtype=float)
        if "Cumulative_Degradation" in X.columns:
            deg = X["Cumulative_Degradation"].to_numpy(dtype=float)
            # soft normalize by train-free rank proxy using values themselves
            if np.nanmax(deg) > np.nanmin(deg):
                score += 0.15 * (deg - np.nanmin(deg)) / (
                    np.nanmax(deg) - np.nanmin(deg) + 1e-9
                )
        score = np.clip(score, 0.0, 1.0)
        return np.column_stack([1 - score, score])
