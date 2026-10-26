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

## Metrics (5-fold GroupKFold by Race · demo train)

Shipped offline demo (~22k synthetic laps, pit rate ≈ 24%). Replace with official Kaggle CSVs for competition-scale runs.

| Model | ROC-AUC | PR-AUC | Recall@top-5% | Recall@top-10% |
| --- | ---: | ---: | ---: | ---: |
| B0 prevalence | 0.500 | 0.239 | 0.046 | 0.101 |
| B1 rule (tyre / late / caution) | 0.671 | 0.373 | 0.093 | 0.189 |
| **Logistic (primary)** | **0.690** | **0.411** | **0.112** | **0.208** |
| HistGradientBoosting | 0.685 | 0.403 | 0.111 | 0.203 |

Means over folds; see `docs/reports/cv_metrics.csv`. Primary model = best **PR-AUC** (tie-break ROC-AUC).

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

# OR offline synthetic demo (no Kaggle account)
python -m src.data.generate_demo

python -m src.models.train   # optional if using committed models/demo/
pytest
streamlit run app/streamlit_app.py
```

A trained demo bundle is committed under `models/demo/` so the triage UI opens without a full retrain.

### Manual Kaggle download

1. https://www.kaggle.com/competitions/playground-series-s6e5/data  
2. Accept rules → Download All  
3. Unzip `train.csv`, `test.csv`, `sample_submission.csv` into `data/raw/`

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

- Demo metrics are on **synthetic** data unless you load official Kaggle CSVs.
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
