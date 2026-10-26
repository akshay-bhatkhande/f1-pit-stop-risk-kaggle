"""Segment error analysis on OOF predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import classification_report_scores


def _bucket_tyre_life(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=[-0.1, 5, 15, 25, 1e9],
        labels=["0-5", "6-15", "16-25", "26+"],
    )


def _bucket_progress(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["early", "mid", "late"],
    )


def _bucket_position(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=[0, 5, 10, 20, 100],
        labels=["P1-5", "P6-10", "P11-20", "P21+"],
    )


def segment_tables(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Build ≥3 slice metric tables."""
    work = frame.copy()
    work["_y"] = y_true
    work["_p"] = y_score
    tables: dict[str, pd.DataFrame] = {}

    # 1) Compound × tyre-life bucket (or compound alone)
    if "Compound" in work.columns and "TyreLife" in work.columns:
        work["tyre_bucket"] = _bucket_tyre_life(work["TyreLife"])
        key = work["Compound"].astype(str) + "|" + work["tyre_bucket"].astype(str)
        tables["compound_tyre_life"] = _agg(work, key)
    elif "Compound" in work.columns:
        tables["compound"] = _agg(work, work["Compound"].astype(str))

    # 2) Race progress phase
    if "RaceProgress" in work.columns:
        work["phase"] = _bucket_progress(work["RaceProgress"])
        tables["race_phase"] = _agg(work, work["phase"].astype(str))

    # 3) Track status or position buckets
    if "TrackStatus" in work.columns:
        tables["track_status"] = _agg(work, work["TrackStatus"].astype(str))
    elif "Position" in work.columns:
        work["pos_bucket"] = _bucket_position(work["Position"])
        tables["position"] = _agg(work, work["pos_bucket"].astype(str))

    return tables


def _agg(work: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    rows = []
    for name, g in work.groupby(key, observed=True):
        if len(g) < 30:
            continue
        scores = classification_report_scores(g["_y"].to_numpy(), g["_p"].to_numpy())
        rows.append(
            {
                "segment": str(name),
                "n": int(len(g)),
                "pit_rate": float(g["_y"].mean()),
                "roc_auc": scores["roc_auc"],
                "pr_auc": scores["pr_auc"],
                "recall@top10%": scores["recall@top10%"],
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)
