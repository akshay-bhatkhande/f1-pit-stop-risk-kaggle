# Decisions

## Problem framing

- **Stakeholder:** race-strategy / ops triage analyst — not a fantasy fan.
- **Question:** P(driver pits on the next lap) given contemporaneous lap state; rank cars for desk review under ~4:1 imbalance.
- **Non-claims:** not live telemetry production; not medal-tier private LB. Official S6E5 CSVs are used for training; CV scores are local GroupKFold, not a competition submission.

## Data path

| Choice | Why |
| --- | --- |
| Primary: Kaggle CLI / local official `playground-series-s6e5` CSVs | Official CC BY 4.0 competition files (~439k train) |
| Fallback: `python -m src.data.generate_demo` | $0 offline clone-and-run when official CSVs are missing |
| Do **not** silently mix original F1 strategy `Normalized_TyreLife` into train | Spec leakage / honesty failure |

### Shipped build status (2026-10-26 retrain)

- Official S6E5 CSVs copied from `Desktop/Job Repo/playground-series-s6e5` → `data/raw/` (CSVs gitignored).
- `DATA_SOURCE.txt` first line = `kaggle_playground_s6e5`.
- Committed artifacts (`models/demo/*`, README metrics, `docs/reports/*`) are from **official train** (439,140 rows, pit rate ≈ 19.9%).
- Provenance: `models/demo/meta.json` (`data_source`, `metrics_are_official_kaggle=true`); Streamlit shows success banner when official.

## Model pick

| Model | Role |
| --- | --- |
| B0 prevalence | Imbalance floor |
| B1 tyre-life / late-progress rule | Beats “I only fitted a booster” narrative |
| Logistic regression (balanced) | Interpretability + PR/ROC baseline |
| **HistGradientBoosting** (sklearn) | Single booster family — CPU-friendly, no LightGBM/XGBoost install friction |

**Primary for Streamlit:** better of logistic vs HistGBM on **PR-AUC** (tie-break ROC-AUC). On official S6E5 CV, **HistGBM** wins both PR-AUC and ROC-AUC and is the shipped primary.

## Features kept / rejected

**Kept (contemporaneous, present in official CSVs):** LapNumber, Position, Stint, TyreLife, RaceProgress, LapTime (+ delta), Cumulative_Degradation, Position_Change, Year, Compound, Driver; engineered `tyre_life_per_progress`, `tyre_life_x_stint`, phase flags.

**Absent on official S6E5 (skipped when missing):** GapToLeader, IntervalToNext, TrackStatus / `track_caution` — demo generator had them; real CSVs do not.

**Rejected / banned:** `PitNextLap` as feature; `Normalized_TyreLife`; `PitStop` as feature; raw `id` as numeric signal; Race one-hot (Race reserved for GroupKFold).

## CV protocol

Prefer **5-fold GroupKFold by Race**. Document StratifiedKFold only if race cardinality is insufficient.

## Segment findings → one guardrail

OOF slices on official data show **late race** and high tyre-life / soft compounds as higher pit-rate regimes with uneven recall@top-k.

**Guardrail shipped:** Streamlit caveats panel calls out late-stint failure modes; `late_race` / phase features retained. No SMOTE, no ensemble stack.

## Cut list ownership (respected)

No Kaggle medal chase, no stacking zoos, no pseudo-labeling, no AutoGluon/DL, no paid APIs, no multi-page SHAP farm, no Streamlit Cloud requirement for DoD.
