# Decisions

## Problem framing

- **Stakeholder:** race-strategy / ops triage analyst — not a fantasy fan.
- **Question:** P(driver pits on the next lap) given contemporaneous lap state; rank cars for desk review under ~4:1 imbalance.
- **Non-claims:** not live telemetry production; not medal-tier private LB; Playground/demo data is synthetic-inspired.

## Data path

| Choice | Why |
| --- | --- |
| Primary: Kaggle CLI `playground-series-s6e5` | Official CC BY 4.0 competition files (~439k train) |
| Fallback: `python -m src.data.generate_demo` | $0 offline clone-and-run when `~/.kaggle/kaggle.json` is missing |
| Do **not** silently mix original F1 strategy `Normalized_TyreLife` into train | Spec leakage / honesty failure |

## Model pick

| Model | Role |
| --- | --- |
| B0 prevalence | Imbalance floor |
| B1 tyre-life / late-progress / caution rule | Beats “I only fitted a booster” narrative |
| Logistic regression (balanced) | Interpretability + PR/ROC baseline |
| **HistGradientBoosting** (sklearn) | Single booster family — CPU-friendly, no LightGBM/XGBoost install friction |

**Primary for Streamlit:** better of logistic vs HistGBM on **PR-AUC** (tie-break ROC-AUC). On the shipped demo CV, logistic edged PR-AUC and is the default primary; HistGBM remains trained and reported.

## Features kept / rejected

**Kept (contemporaneous):** LapNumber, Position, Stint, TyreLife, RaceProgress, LapTime (+ delta), Cumulative_Degradation, Position_Change, gaps/intervals, TrackStatus, Compound, Driver; engineered `tyre_life_per_progress`, `tyre_life_x_stint`, phase flags, `track_caution`.

**Rejected / banned:** `PitNextLap` as feature; `Normalized_TyreLife`; `PitStop` as feature; raw `id` as numeric signal; Race one-hot (Race reserved for GroupKFold).

## CV protocol

Prefer **5-fold GroupKFold by Race**. Document StratifiedKFold only if race cardinality is insufficient.

## Segment findings → one guardrail

OOF slices on demo data showed **late race** and **caution TrackStatus (4/6/7)** have higher pit rates and weaker recall@top-10% relative to green-flag early/mid laps.

**Guardrail shipped:** Streamlit caveats panel calls out late-stint and caution-regime failure modes; `track_caution` + `late_race` features retained so the desk score is not blind to those regimes. No SMOTE, no ensemble stack.

## Cut list ownership (respected)

No Kaggle medal chase, no stacking zoos, no pseudo-labeling, no AutoGluon/DL, no paid APIs, no multi-page SHAP farm, no Streamlit Cloud requirement for DoD.
