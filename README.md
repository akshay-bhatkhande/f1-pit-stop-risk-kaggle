# F1 Pit-Stop Risk Lab

**Ops triage under imbalance** — baselines, leakage hygiene, ROC + PR + recall@top-k, and a Streamlit desk — not another Kaggle blend.

> Playground / demo data is **synthetic** (inspired by an F1 strategy dataset). Treat this as ML literacy + framing for a strategy-desk analyst, **not** race-engineering truth or live telemetry.

| | |
| --- | --- |
| **Business question** | Given current lap state, what is P(driver pits next lap)? Which cars should the desk flag for near-term pit attention? |
| **Task** | Binary classification · target `PitNextLap` |
| **Stack** | pandas · scikit-learn (logistic + HistGradientBoosting) · Streamlit · pytest |
| **Cost** | $0 paid APIs |
| **Competition** | [Predicting F1 Pit Stops — Playground S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) (closed; portfolio use only — **no medal chase**) |
| **License (competition data)** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

---

## Metrics (5-fold GroupKFold by Race)

> **Honesty label:** these numbers are from **SYNTHETIC DEMO** data (`python -m src.data.generate_demo`, ~21.7k train rows, pit rate ≈ 24%).  
> They are **not** official Kaggle Playground S6E5 / competition-scale results. Build machine had no `~/.kaggle/kaggle.json`.  
> After you drop in real CSVs and retrain, replace this table from `docs/reports/cv_metrics.csv`.

| Model | ROC-AUC | PR-AUC | Recall@top-5% | Recall@top-10% |
| --- | ---: | ---: | ---: | ---: |
| B0 prevalence | 0.500 | 0.239 | 0.046 | 0.101 |
| B1 rule (tyre / late / caution) | 0.671 | 0.373 | 0.093 | 0.189 |
| **Logistic (primary)** | **0.690** | **0.411** | **0.112** | **0.208** |
| HistGradientBoosting | 0.685 | 0.403 | 0.111 | 0.203 |

Means over folds; see `docs/reports/cv_metrics.csv`. Primary model = best **PR-AUC** (tie-break ROC-AUC). Provenance: `models/demo/meta.json` → `data_source=synthetic_demo`.

**Recall@top-k** = of true pits, how many fall in the top *k%* of scored eval rows (ops “how many alerts can we review?”).

---

## Quick start

```bash
git clone https://github.com/akshay-bhatkhande/f1-pit-stop-risk-lab.git
cd f1-pit-stop-risk-lab
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Data — preferred (Kaggle token in ~/.kaggle/kaggle.json, chmod 600)
bash scripts/download_data.sh

# OR offline synthetic demo (no Kaggle account) — metrics will be demo-only
python -m src.data.generate_demo

python -m src.models.train   # required after swapping to real Kaggle CSVs
pytest
streamlit run app/streamlit_app.py
```

A trained **synthetic-demo** bundle is committed under `models/demo/` so the triage UI opens without a full retrain. The UI banners this clearly.

### Owner: swap synthetic → real Kaggle CSVs

1. Create a free Kaggle account; accept [S6E5 rules](https://www.kaggle.com/competitions/playground-series-s6e5/rules).  
2. Account → Settings → API → Create New Token → save as `~/.kaggle/kaggle.json` → `chmod 600 ~/.kaggle/kaggle.json`.  
3. From repo root: `FORCE=1 bash scripts/download_data.sh`  
   (or manually unzip `train.csv` / `test.csv` / `sample_submission.csv` into `data/raw/` and write `kaggle_playground_s6e5` as the first line of `data/raw/DATA_SOURCE.txt`).  
4. `python -m src.models.train` then refresh Streamlit / update the README metrics table from `docs/reports/cv_metrics.csv`.

### Manual Kaggle download

1. https://www.kaggle.com/competitions/playground-series-s6e5/data  
2. Accept rules → Download All  
3. Unzip `train.csv`, `test.csv`, `sample_submission.csv` into `data/raw/`  
4. Retrain (step 4 above) — do not keep synthetic metrics after swapping files.

---

## What’s in the box

- **Baselines** before models (prevalence + simple tyre/progress/caution rule)
- **Logistic + one booster** (HistGradientBoosting) — no ensembles
- **Leakage checklist** — `docs/leakage-checklist.md`
- **Segment error analysis** — compound×tyre life, race phase, track status (`docs/reports/slice_*.csv`)
- **Streamlit triage** — top-k / threshold budget → attention list + recall

```text
app/streamlit_app.py      # desk UI
src/data/                 # load + demo generator
src/features/             # contemporaneous features
src/models/               # baselines + train CV
src/eval/                 # metrics + slices
docs/                     # decisions, build-log, leakage, reports
tests/                    # smoke train, metrics, banned columns
scripts/download_data.sh
```

---

## Limitations

- **Shipped metrics/models are synthetic-demo only** until the owner downloads official S6E5 CSVs and retrains (see above).
- Not a production pit-wall system; no real-time telemetry.
- Late-race and caution regimes are harder — see caveats in the UI and `docs/decisions.md`.
- Accuracy is intentionally **not** a hero metric under imbalance.

---

## Citation

Competition: Kaggle Playground Series S6E5 — Predicting F1 Pit Stops.  
Data inspired by a public F1 strategy dataset; `Normalized_TyreLife` intentionally removed by Kaggle as trivial.

---

## Docs

- [decisions.md](docs/decisions.md) — model pick, rejected features, segment guardrail  
- [leakage-checklist.md](docs/leakage-checklist.md)  
- [build-log.md](docs/build-log.md) — AI-assisted steps + human-owned calls  
