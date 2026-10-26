"""Provenance helpers must not silently claim Kaggle for demo CSVs."""

import pandas as pd

from src.data.generate_demo import generate
from src.data.load import (
    SOURCE_KAGGLE,
    SOURCE_SYNTHETIC,
    infer_data_source,
    read_data_source_marker,
    write_data_source,
)


def test_generate_demo_writes_synthetic_marker(tmp_path):
    generate(n_train_races=2, n_test_races=1, seed=3, out_dir=tmp_path)
    assert read_data_source_marker(tmp_path) == SOURCE_SYNTHETIC
    train = pd.read_csv(tmp_path / "train.csv")
    assert infer_data_source(train, directory=tmp_path) == SOURCE_SYNTHETIC


def test_row_count_heuristic_marks_large_as_kaggle(tmp_path):
    write_data_source(SOURCE_KAGGLE, directory=tmp_path)
    assert infer_data_source(directory=tmp_path) == SOURCE_KAGGLE


def test_small_frame_without_marker_is_synthetic(tmp_path):
    df = pd.DataFrame({"Race": ["R01_Y2024"] * 100, "PitNextLap": [0] * 80 + [1] * 20})
    assert infer_data_source(df, directory=tmp_path) == SOURCE_SYNTHETIC
