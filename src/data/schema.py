"""Column contracts and banned / key fields."""

from __future__ import annotations

TARGET = "PitNextLap"

# Never use as model features
BANNED_COLUMNS = frozenset(
    {
        TARGET,
        "Normalized_TyreLife",  # intentionally removed by Kaggle; trivial if reintroduced
        "PitStop",  # current-lap pit outcome can leak next-lap structure depending on join
    }
)

# Identifiers — keys / grouping only, not numeric signal
ID_COLUMNS = frozenset({"id", "Id", "ID"})

# Preferred race grouping keys (first present wins)
GROUP_CANDIDATES = ("Race", "RaceID", "race", "race_id")

# Categorical candidates (used if present)
CATEGORICAL_CANDIDATES = (
    "Compound",
    "Driver",
    "DriverID",
    "TrackStatus",
    "Race",
    "RaceID",
)

# Numeric candidates (used if present)
NUMERIC_CANDIDATES = (
    "LapNumber",
    "Position",
    "Stint",
    "TyreLife",
    "RaceProgress",
    "LapTime",
    "LapTime_s",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "Position_Change",
    "GapToLeader",
    "IntervalToNext",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Year",
)

# Expected Playground-inspired column set for demo generator
DEMO_COLUMNS = [
    "id",
    "Year",
    "Race",
    "Driver",
    "LapNumber",
    "Position",
    "Stint",
    "Compound",
    "TyreLife",
    "RaceProgress",
    "LapTime",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "Position_Change",
    "TrackStatus",
    "GapToLeader",
    "IntervalToNext",
    TARGET,
]
