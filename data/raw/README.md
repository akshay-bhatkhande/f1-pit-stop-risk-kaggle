# Raw competition data

**Current local status:** official S6E5 CSVs present; `DATA_SOURCE.txt` first line = `kaggle_playground_s6e5` (CSVs are gitignored).

Place Kaggle Playground Series S6E5 files here:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Preferred path

```bash
# from repo root, after placing ~/.kaggle/kaggle.json (chmod 600)
bash scripts/download_data.sh
```

## Manual fallback

1. Open https://www.kaggle.com/competitions/playground-series-s6e5/data
2. Accept competition rules → **Download All**
3. Unzip CSVs into this folder

## Offline demo (no Kaggle token)

```bash
python -m src.data.generate_demo
```

This writes a smaller synthetic `train.csv` / `test.csv` shaped like the Playground schema so training and Streamlit work without API credentials. It also writes `DATA_SOURCE.txt` with `synthetic_demo`.

**Do not treat demo metrics as Kaggle results.** After placing real CSVs:

```bash
FORCE=1 bash scripts/download_data.sh   # or manual unzip into this folder
# ensure DATA_SOURCE.txt first line is: kaggle_playground_s6e5
python -m src.models.train
```

**License:** competition data is CC BY 4.0 per Kaggle. Do not commit raw CSVs.
