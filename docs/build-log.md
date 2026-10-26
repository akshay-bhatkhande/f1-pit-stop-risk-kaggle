# Build log

AI-assisted scaffold + human-owned metric / leakage calls for `f1-pit-stop-risk-lab`.

## Timeline (2026-10-26)

1. **Spec lock** — followed frozen `/docs/specs/f1-pit-stop-risk-lab-spec.md` (no scope invention).
2. **Scaffold** — repo layout, `.gitignore`, `requirements.txt`, download script, schema module.
3. **Data** — Kaggle download path scripted; no local `kaggle.json` on build machine → synthetic demo generator matching Playground-inspired columns (CC BY framing; honesty in README).
4. **Baselines → models** — B0/B1, logistic, HistGradientBoosting; GroupKFold by Race; metrics ROC-AUC / PR-AUC / recall@top-5% / recall@top-10%.
5. **Leakage** — banned-column tests + `docs/leakage-checklist.md`.
6. **Segments** — compound×tyre-life, race phase, track status CSVs under `docs/reports/`; guardrail noted in decisions + UI caveats.
7. **Artifact** — `models/demo/triage_bundle.joblib` + OOF parquet for Streamlit.
8. **UI** — `app/streamlit_app.py` attention list + alert budget controls.
9. **Verify** — `pytest` green; Streamlit smoke; demo media captured for portfolio store.
10. **Review (2026-10-26)** — still no Kaggle token on machine; reinforced synthetic labeling (DATA_SOURCE marker, meta flags, UI banner, README owner steps, download script FORCE overwrite). Did **not** retrain on official CSVs (unavailable).
11. **Official retrain (2026-10-26)** — Akshay placed S6E5 zip contents at `Desktop/Job Repo/playground-series-s6e5` (~439k train). Copied into `data/raw/`, marked `kaggle_playground_s6e5`, retrained full CV. Primary flipped to **HistGBM** (ROC-AUC ≈ 0.928, PR-AUC ≈ 0.724). README / decisions / meta / UI provenance updated for real data. Comparison vs winners written in portfolio store.

## Human-owned calls

- Primary model selection rule: **PR-AUC first**, not ROC vanity.
- Accuracy banned as hero metric.
- Demo vs Kaggle data clearly labeled; never claim medal skill / private LB.
- Cut list enforced (no ensembles / SMOTE headline / paid APIs).

## Commands used

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh   # or: python -m src.data.generate_demo
python -m src.models.train
pytest
streamlit run app/streamlit_app.py
```
