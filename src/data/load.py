"""Load and validate raw CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import TARGET

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Official Playground S6E5 train is ~439k rows; demo generator is ~20–30k.
KAGGLE_TRAIN_ROWS_MIN = 300_000
SOURCE_SYNTHETIC = "synthetic_demo"
SOURCE_KAGGLE = "kaggle_playground_s6e5"
SOURCE_UNKNOWN = "unknown"


def raw_paths() -> dict[str, Path]:
    return {
        "train": RAW_DIR / "train.csv",
        "test": RAW_DIR / "test.csv",
        "sample": RAW_DIR / "sample_submission.csv",
        "source": RAW_DIR / "DATA_SOURCE.txt",
    }


def data_ready() -> bool:
    p = raw_paths()
    return p["train"].exists() and p["test"].exists()


def write_data_source(
    kind: str, note: str = "", directory: Path | None = None
) -> Path:
    """Persist provenance for UI / README honesty (never commit raw CSVs)."""
    path = (directory or RAW_DIR) / "DATA_SOURCE.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = kind.strip()
    if note:
        body = f"{body}\n{note.strip()}\n"
    else:
        body = f"{body}\n"
    path.write_text(body)
    return path


def read_data_source_marker(directory: Path | None = None) -> str | None:
    path = (directory or RAW_DIR) / "DATA_SOURCE.txt"
    if not path.exists():
        return None
    first = path.read_text().strip().splitlines()
    return first[0].strip() if first else None


def infer_data_source(
    train: pd.DataFrame | None = None, directory: Path | None = None
) -> str:
    """
    Resolve whether metrics/UI should claim Kaggle or synthetic demo.

    Order: explicit DATA_SOURCE.txt → row-count heuristic → race-name heuristic.
    """
    marked = read_data_source_marker(directory)
    if marked in {SOURCE_SYNTHETIC, SOURCE_KAGGLE}:
        return marked

    df = train
    train_path = (directory or RAW_DIR) / "train.csv"
    if df is None and train_path.exists():
        # Cheap: only need shape + Race sample for heuristics
        df = pd.read_csv(train_path, usecols=lambda c: c in {"Race", TARGET})

    if df is None or len(df) == 0:
        return SOURCE_UNKNOWN

    n = len(df)
    if n >= KAGGLE_TRAIN_ROWS_MIN:
        return SOURCE_KAGGLE

    if "Race" in df.columns:
        races = df["Race"].astype(str)
        if races.str.contains(r"_TEST|_Y20", regex=True, na=False).any():
            return SOURCE_SYNTHETIC

    if n < KAGGLE_TRAIN_ROWS_MIN:
        return SOURCE_SYNTHETIC
    return SOURCE_UNKNOWN


def data_source_label(kind: str | None = None) -> str:
    kind = kind or infer_data_source()
    if kind == SOURCE_KAGGLE:
        return "Kaggle Playground S6E5 (official CSVs)"
    if kind == SOURCE_SYNTHETIC:
        return "SYNTHETIC DEMO (not official Kaggle CSVs)"
    return "unknown data source — verify data/raw/"

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
    source = infer_data_source(df)
    out: dict = {
        "name": name,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "data_source": source,
        "data_source_label": data_source_label(source),
    }
    if TARGET in df.columns:
        rate = float(df[TARGET].mean())
        out["pit_rate"] = rate
        out["class_counts"] = df[TARGET].value_counts().to_dict()
    missing = df.isna().sum()
    out["missing_nonzero"] = {c: int(v) for c, v in missing.items() if v > 0}
    out["n_missing_cells"] = int(missing.sum())
    return out
