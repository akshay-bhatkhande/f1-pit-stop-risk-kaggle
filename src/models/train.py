"""Train baselines + logistic + HistGradientBoosting with group-aware CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.load import load_train, summarize_frame
from src.eval.metrics import classification_report_scores, mean_std_table
from src.eval.slices import segment_tables
from src.features.build import engineer, make_preprocessor, prepare_xy
from src.models.baselines import PrevalenceBaseline, RuleBaseline

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "docs" / "reports"
RANDOM_STATE = 42


def _cv_splitter(groups: np.ndarray | None, n_splits: int = 5):
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return GroupKFold(n_splits=n_splits), "GroupKFold(Race)"
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    ), "StratifiedKFold (group cardinality insufficient)"


def _oof_baselines(X: pd.DataFrame, y: np.ndarray, splits) -> dict:
    results = {}
    for name, Cls in (("B0_prevalence", PrevalenceBaseline), ("B1_rule", RuleBaseline)):
        oof = np.zeros(len(y), dtype=float)
        fold_scores = []
        for train_idx, val_idx in splits:
            model = Cls().fit(X.iloc[train_idx], y[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            oof[val_idx] = proba
            fold_scores.append(classification_report_scores(y[val_idx], proba))
        results[name] = {
            "fold_metrics": fold_scores,
            "summary": mean_std_table(fold_scores),
            "oof": oof,
            "model": Cls().fit(X, y),
        }
    return results


def _oof_sklearn_pipeline(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    plan,
    splits,
) -> dict:
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for train_idx, val_idx in splits:
        pre = make_preprocessor(plan)
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        # Fresh estimator each fold
        if name == "logistic":
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                max_iter=200,
                random_state=RANDOM_STATE,
            )
        pipe = Pipeline([("pre", make_preprocessor(plan)), ("clf", clf)])
        pipe.fit(X.iloc[train_idx], y[train_idx])
        proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        fold_scores.append(classification_report_scores(y[val_idx], proba))
    # Fit final on all data
    if name == "logistic":
        final_clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        final_clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
    final = Pipeline([("pre", make_preprocessor(plan)), ("clf", final_clf)])
    final.fit(X, y)
    return {
        "fold_metrics": fold_scores,
        "summary": mean_std_table(fold_scores),
        "oof": oof,
        "model": final,
    }


def run_training(max_rows: int | None = None) -> dict:
    df = load_train()
    summary = summarize_frame(df)
    print(
        f"Loaded train: {summary['n_rows']:,} rows | "
        f"pit_rate={summary.get('pit_rate', float('nan')):.3f} | "
        f"missing_cells={summary['n_missing_cells']}"
    )
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows for this run")

    work = engineer(df)
    X, y, plan = prepare_xy(df)
    groups = None
    if plan.group_col and plan.group_col in work.columns:
        groups = work[plan.group_col].astype(str).to_numpy()

    splitter, split_name = _cv_splitter(groups)
    if groups is not None and split_name.startswith("Group"):
        splits = list(splitter.split(X, y, groups))
    else:
        splits = list(splitter.split(X, y))
    print(f"CV protocol: {split_name} | features={plan.feature_names}")

    results = {}
    results.update(_oof_baselines(X, y, splits))
    results["logistic"] = _oof_sklearn_pipeline(
        "logistic", None, X, y, plan, splits
    )
    results["hist_gbm"] = _oof_sklearn_pipeline(
        "hist_gbm", None, X, y, plan, splits
    )

    # Primary model: better PR-AUC, tie-break ROC-AUC
    def key_metric(name: str) -> tuple[float, float]:
        s = results[name]["summary"]
        return (s["pr_auc"]["mean"], s["roc_auc"]["mean"])

    primary = max(("logistic", "hist_gbm"), key=key_metric)
    print(f"Primary model for Streamlit: {primary}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    demo_dir = MODELS_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "primary": primary,
        "plan": plan,
        "split_name": split_name,
        "feature_names": plan.feature_names,
        "data_summary": summary,
    }
    joblib.dump(
        {
            **artifact,
            "model": results[primary]["model"],
            "logistic": results["logistic"]["model"],
            "baselines": {
                "B0": results["B0_prevalence"]["model"],
                "B1": results["B1_rule"]["model"],
            },
        },
        demo_dir / "triage_bundle.joblib",
    )

    # Metrics markdown + JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for name in ("B0_prevalence", "B1_rule", "logistic", "hist_gbm"):
        s = results[name]["summary"]
        metrics_rows.append(
            {
                "model": name,
                "roc_auc_mean": s["roc_auc"]["mean"],
                "roc_auc_std": s["roc_auc"]["std"],
                "pr_auc_mean": s["pr_auc"]["mean"],
                "pr_auc_std": s["pr_auc"]["std"],
                "recall@top5%_mean": s["recall@top5%"]["mean"],
                "recall@top10%_mean": s["recall@top10%"]["mean"],
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(REPORTS_DIR / "cv_metrics.csv", index=False)

    # OOF eval table for UI
    oof_primary = results[primary]["oof"]
    eval_table = work.copy()
    eval_table["y_true"] = y
    eval_table["y_score"] = oof_primary
    keep_cols = [
        c
        for c in [
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
            "TrackStatus",
            "y_true",
            "y_score",
        ]
        if c in eval_table.columns
    ]
    eval_path = demo_dir / "oof_eval.parquet"
    try:
        eval_table[keep_cols].to_parquet(eval_path, index=False)
    except Exception:
        eval_table[keep_cols].to_csv(demo_dir / "oof_eval.csv", index=False)

    # Segment analysis
    slices = segment_tables(work, y, oof_primary)
    for sname, sdf in slices.items():
        sdf.to_csv(REPORTS_DIR / f"slice_{sname}.csv", index=False)

    meta = {
        "primary": primary,
        "split": split_name,
        "features": plan.feature_names,
        "engineered": plan.engineered,
        "metrics": metrics_rows,
        "n_rows_trained": int(len(df)),
        "pit_rate": summary.get("pit_rate"),
    }
    (demo_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (REPORTS_DIR / "cv_metrics.json").write_text(json.dumps(meta, indent=2))

    print(metrics_df.to_string(index=False))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional subsample for smoke/debug runs",
    )
    args = parser.parse_args()
    run_training(max_rows=args.max_rows)


if __name__ == "__main__":
    main()
