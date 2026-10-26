"""Feature matrix construction — contemporaneous fields only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import (
    BANNED_COLUMNS,
    CATEGORICAL_CANDIDATES,
    GROUP_CANDIDATES,
    ID_COLUMNS,
    NUMERIC_CANDIDATES,
    TARGET,
)


@dataclass
class FeaturePlan:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    group_col: str | None = None
    engineered: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + self.categorical


def resolve_group_col(columns: pd.Index | list[str]) -> str | None:
    cols = set(columns)
    for c in GROUP_CANDIDATES:
        if c in cols:
            return c
    return None


def _normalize_laptime_col(df: pd.DataFrame) -> pd.DataFrame:
    """Unify LapTime column naming variants."""
    out = df.copy()
    if "LapTime" not in out.columns:
        for alt in ("LapTime_s", "LapTime (s)"):
            if alt in out.columns:
                out = out.rename(columns={alt: "LapTime"})
                break
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add safe contemporaneous proxies (no future info)."""
    out = _normalize_laptime_col(df)
    # Progress-normalized tyre age proxy (not Normalized_TyreLife from external file)
    if "TyreLife" in out.columns and "RaceProgress" in out.columns:
        prog = out["RaceProgress"].replace(0, np.nan)
        out["tyre_life_per_progress"] = out["TyreLife"] / prog.clip(lower=0.01)
        out["tyre_life_per_progress"] = out["tyre_life_per_progress"].fillna(0.0)
    if "TyreLife" in out.columns and "Stint" in out.columns:
        out["tyre_life_x_stint"] = out["TyreLife"] * out["Stint"]
    if "RaceProgress" in out.columns:
        out["late_race"] = (out["RaceProgress"] >= 0.66).astype(int)
        out["mid_race"] = (
            (out["RaceProgress"] >= 0.33) & (out["RaceProgress"] < 0.66)
        ).astype(int)
    if "TrackStatus" in out.columns:
        out["track_caution"] = (
            out["TrackStatus"].astype(str).isin(["4", "5", "6", "7", "SC", "VSC"])
        ).astype(int)
    return out


def plan_features(df: pd.DataFrame) -> FeaturePlan:
    engineered_cols = [
        c
        for c in (
            "tyre_life_per_progress",
            "tyre_life_x_stint",
            "late_race",
            "mid_race",
            "track_caution",
        )
        if c in df.columns
    ]
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    numeric += [c for c in engineered_cols if c not in numeric]
    # Drop ID-like numerics if they slipped in
    numeric = [c for c in numeric if c not in ID_COLUMNS and c not in BANNED_COLUMNS]

    categorical = [
        c
        for c in CATEGORICAL_CANDIDATES
        if c in df.columns and c not in BANNED_COLUMNS and c not in ID_COLUMNS
    ]
    # Prefer keeping Race as group key only — exclude from features if used as group
    group = resolve_group_col(df.columns)
    if group and group in categorical:
        categorical = [c for c in categorical if c != group]

    # Driver as categorical is OK for trees/logistic with one-hot; fine at demo scale
    return FeaturePlan(
        numeric=numeric,
        categorical=categorical,
        group_col=group,
        engineered=engineered_cols,
    )


def make_preprocessor(plan: FeaturePlan) -> ColumnTransformer:
    transformers = []
    if plan.numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                plan.numeric,
            )
        )
    if plan.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                plan.categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns found")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def assert_no_banned(feature_names: list[str]) -> None:
    banned_hit = [c for c in feature_names if c in BANNED_COLUMNS or c == TARGET]
    if banned_hit:
        raise ValueError(f"Banned columns in feature list: {banned_hit}")


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, FeaturePlan]:
    work = engineer(df)
    plan = plan_features(work)
    assert_no_banned(plan.feature_names)
    cols = plan.feature_names
    X = work[cols].copy()
    y = work[TARGET].astype(int).to_numpy()
    return X, y, plan
