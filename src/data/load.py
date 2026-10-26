"""Load and validate raw CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import TARGET

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"


def raw_paths() -> dict[str, Path]:
    return {
        "train": RAW_DIR / "train.csv",
        "test": RAW_DIR / "test.csv",
        "sample": RAW_DIR / "sample_submission.csv",
    }


def data_ready() -> bool:
    p = raw_paths()
    return p["train"].exists() and p["test"].exists()


def load_train(path: Path | None = None) -> pd.DataFrame:
    train_path = path or raw_paths()["train"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run: bash scripts/download_data.sh "
            "or python -m src.data.generate_demo"
        )
    df = pd.read_csv(train_path)
    if TARGET not in df.columns:
        raise ValueError(f"Train CSV missing target column '{TARGET}'")
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    test_path = path or raw_paths()["test"]
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")
    return pd.read_csv(test_path)


def summarize_frame(df: pd.DataFrame, name: str = "train") -> dict:
    """Return imbalance + missingness summary for logging / docs."""
    out: dict = {
        "name": name,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }
    if TARGET in df.columns:
        rate = float(df[TARGET].mean())
        out["pit_rate"] = rate
        out["class_counts"] = df[TARGET].value_counts().to_dict()
    missing = df.isna().sum()
    out["missing_nonzero"] = {c: int(v) for c, v in missing.items() if v > 0}
    out["n_missing_cells"] = int(missing.sum())
    return out
