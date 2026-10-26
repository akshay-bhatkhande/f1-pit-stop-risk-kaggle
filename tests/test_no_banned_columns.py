"""Feature plan must exclude target and banned trivial columns."""

import pandas as pd

from src.data.schema import BANNED_COLUMNS, TARGET
from src.features.build import assert_no_banned, engineer, plan_features


def test_banned_not_in_plan():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "Race": ["A", "A", "B", "B"],
            "Driver": ["D1", "D2", "D1", "D2"],
            "LapNumber": [1, 2, 1, 2],
            "Position": [1, 2, 3, 4],
            "Stint": [1, 1, 1, 1],
            "Compound": ["SOFT", "SOFT", "HARD", "HARD"],
            "TyreLife": [1, 2, 1, 2],
            "RaceProgress": [0.1, 0.2, 0.1, 0.2],
            "LapTime": [90.0, 90.1, 91.0, 91.2],
            "LapTime_Delta": [0.0, 0.1, 0.0, 0.2],
            "Cumulative_Degradation": [0.1, 0.2, 0.1, 0.2],
            "Position_Change": [0, 0, 0, 0],
            "TrackStatus": ["1", "1", "1", "4"],
            "Normalized_TyreLife": [0.1, 0.2, 0.1, 0.2],  # must be ignored
            TARGET: [0, 1, 0, 0],
        }
    )
    work = engineer(df)
    plan = plan_features(work)
    assert TARGET not in plan.feature_names
    assert "Normalized_TyreLife" not in plan.feature_names
    assert "id" not in plan.feature_names
    for b in BANNED_COLUMNS:
        assert b not in plan.feature_names
    assert_no_banned(plan.feature_names)


def test_assert_no_banned_raises():
    try:
        assert_no_banned([TARGET, "TyreLife"])
        raised = False
    except ValueError:
        raised = True
    assert raised
