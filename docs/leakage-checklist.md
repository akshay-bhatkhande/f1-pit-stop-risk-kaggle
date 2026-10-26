# Leakage checklist

Status after Build Agent 2 ship (demo + Kaggle-compatible pipeline).

| Check | Status | Notes |
| --- | --- | --- |
| Target `PitNextLap` never used as a feature | ✅ | Excluded via `BANNED_COLUMNS` / `assert_no_banned`; covered by `tests/test_no_banned_columns.py` |
| No future-lap fields (post-pit outcomes, next-lap times) | ✅ | Feature plan uses contemporaneous race-state fields only; `PitStop` banned |
| IDs handled as keys / categoricals with group-aware CV — not numeric signal | ✅ | `id` excluded; `Race` used for `GroupKFold` and not one-hot’d into the matrix |
| No competition-removed `Normalized_TyreLife` from an external file as a silent train column | ✅ | Banned even if present in a joined CSV; progress-normalized tyre proxy is engineered locally as `tyre_life_per_progress` |
| Train/eval split does not leak same-lap duplicates; prefer GroupKFold by Race | ✅ | Default `GroupKFold(Race)` when ≥5 races; else StratifiedKFold with limitation logged in `meta.json` |
| Any target encoding fitted inside CV folds only | ✅ | No target encoding used; one-hot + scaling fit inside each fold via sklearn `Pipeline` |
| Test CSV labels never invented from public notebooks for “training” | ✅ | Test used only for schema/smoke; training uses labeled train only |

**Honesty note:** Shipped metrics/models (2026-10-26) use **official** Playground S6E5 CSVs (`DATA_SOURCE.txt` → `kaggle_playground_s6e5`, ~439k train). Offline clone-and-run can still fall back to `python -m src.data.generate_demo` when CSVs are absent — that path must keep the synthetic banner and must not overwrite the README’s official metrics table without a retrain.
