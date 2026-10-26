"""Tiny subsample trains without crash and yields finite probabilities."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.generate_demo import generate
from src.features.build import make_preprocessor, prepare_xy


def test_smoke_logistic(tmp_path):
    generate(n_train_races=2, n_test_races=1, seed=7, out_dir=tmp_path)
    df = pd.read_csv(tmp_path / "train.csv")
    assert len(df) > 50
    X, y, plan = prepare_xy(df)
    pipe = Pipeline(
        [
            ("pre", make_preprocessor(plan)),
            (
                "clf",
                LogisticRegression(max_iter=400, class_weight="balanced", random_state=0),
            ),
        ]
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert len(proba) == len(y)
    assert np.isfinite(proba).all()
    assert proba.min() >= 0.0 and proba.max() <= 1.0
