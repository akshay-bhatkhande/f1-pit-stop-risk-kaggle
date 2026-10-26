"""Synthetic Playground-shaped CSVs for offline clone-and-run (no Kaggle token)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import SOURCE_SYNTHETIC, write_data_source
from src.data.schema import DEMO_COLUMNS, TARGET

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
TRACK_STATUS = ["1", "2", "4", "6", "7"]  # green / yellow / SC-ish codes
RACES = [f"R{i:02d}" for i in range(1, 25)]
DRIVERS = [f"D{i:02d}" for i in range(1, 21)]


def _simulate_race(
    rng: np.random.Generator,
    race: str,
    year: int,
    n_laps: int,
    start_id: int,
) -> pd.DataFrame:
    rows = []
    rid = start_id
    for driver in DRIVERS:
        stint = 1
        tyre_life = 0
        compound = rng.choice(COMPOUNDS[:3], p=[0.35, 0.40, 0.25])
        base_lap = float(rng.uniform(78.0, 95.0))
        position = int(rng.integers(1, 21))
        prev_lap_time = base_lap
        cum_deg = 0.0
        for lap in range(1, n_laps + 1):
            tyre_life += 1
            progress = lap / n_laps
            # Degradation rises with tyre life; soft compounds wear faster
            wear_rate = {"SOFT": 0.055, "MEDIUM": 0.035, "HARD": 0.022}.get(
                compound, 0.04
            )
            cum_deg += wear_rate * (1.0 + 0.15 * tyre_life)
            lap_time = base_lap + cum_deg * 0.12 + float(rng.normal(0, 0.25))
            delta = lap_time - prev_lap_time
            prev_lap_time = lap_time
            track = rng.choice(TRACK_STATUS, p=[0.82, 0.08, 0.05, 0.03, 0.02])
            gap = max(0.0, (position - 1) * float(rng.uniform(0.4, 1.8)))
            interval = float(rng.exponential(1.2))
            pos_change = int(rng.integers(-1, 2))
            position = int(np.clip(position + pos_change, 1, 20))

            # Pit probability: high tyre life, late race, soft compound, SC-ish
            # Tuned toward ~20% positive rate (Playground-like imbalance)
            logit = -2.55
            logit += 0.14 * tyre_life
            logit += 2.2 * max(0.0, progress - 0.50)
            logit += {"SOFT": 0.65, "MEDIUM": 0.20, "HARD": -0.15}.get(compound, 0.1)
            if track in {"4", "6", "7"}:
                logit += 1.0
            if stint >= 3 and progress > 0.7:
                logit += 0.5
            p_pit = 1.0 / (1.0 + np.exp(-logit))
            pit_next = int(rng.random() < p_pit)

            rows.append(
                {
                    "id": rid,
                    "Year": year,
                    "Race": race,
                    "Driver": driver,
                    "LapNumber": lap,
                    "Position": position,
                    "Stint": stint,
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "RaceProgress": round(progress, 4),
                    "LapTime": round(lap_time, 3),
                    "LapTime_Delta": round(delta, 3),
                    "Cumulative_Degradation": round(cum_deg, 4),
                    "Position_Change": pos_change,
                    "TrackStatus": track,
                    "GapToLeader": round(gap, 3),
                    "IntervalToNext": round(interval, 3),
                    TARGET: pit_next,
                }
            )
            rid += 1

            if pit_next and lap < n_laps:
                # Start next stint on following lap
                stint += 1
                tyre_life = 0
                cum_deg = 0.0
                compound = rng.choice(COMPOUNDS[:3], p=[0.30, 0.45, 0.25])

    return pd.DataFrame(rows)


def generate(
    n_train_races: int = 18,
    n_test_races: int = 6,
    seed: int = 42,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    out = out_dir or RAW_DIR
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    frames = []
    next_id = 0
    for i in range(n_train_races):
        race = RACES[i % len(RACES)]
        year = 2024 + (i // len(RACES))
        n_laps = int(rng.integers(50, 71))
        part = _simulate_race(rng, f"{race}_Y{year}", year, n_laps, next_id)
        next_id = int(part["id"].max()) + 1
        frames.append(part)
    train = pd.concat(frames, ignore_index=True)

    test_frames = []
    for j in range(n_test_races):
        race = RACES[(n_train_races + j) % len(RACES)]
        year = 2025
        n_laps = int(rng.integers(50, 71))
        part = _simulate_race(rng, f"{race}_TEST{j}", year, n_laps, next_id)
        next_id = int(part["id"].max()) + 1
        test_frames.append(part)
    test = pd.concat(test_frames, ignore_index=True)
    test = test.drop(columns=[TARGET])

    assert list(train.columns) == DEMO_COLUMNS
    train_path = out / "train.csv"
    test_path = out / "test.csv"
    sample_path = out / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    pd.DataFrame({"id": test["id"], TARGET: 0.0}).to_csv(sample_path, index=False)
    write_data_source(
        SOURCE_SYNTHETIC,
        note=(
            f"rows_train={len(train)} rows_test={len(test)} "
            "NOT official Kaggle Playground S6E5. "
            "Replace via: bash scripts/download_data.sh "
            "or manual unzip into data/raw/ then retrain."
        ),
        directory=out,
    )

    rate = float(train[TARGET].mean())
    print(f"Wrote {train_path} ({len(train):,} rows, pit_rate={rate:.3f})")
    print(f"Wrote {test_path} ({len(test):,} rows)")
    print("NOTE: SYNTHETIC DEMO DATA — metrics are NOT official Kaggle results.")
    print("      Replace with Kaggle CSVs (scripts/download_data.sh) then retrain.")
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic F1 pit demo CSVs")
    parser.add_argument("--train-races", type=int, default=18)
    parser.add_argument("--test-races", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(n_train_races=args.train_races, n_test_races=args.test_races, seed=args.seed)


if __name__ == "__main__":
    main()
