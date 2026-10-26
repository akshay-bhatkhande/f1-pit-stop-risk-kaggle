# Raw competition data

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

This writes a smaller synthetic `train.csv` / `test.csv` shaped like the Playground schema so training and Streamlit work without API credentials. Real competition CSVs are preferred when available.

**License:** competition data is CC BY 4.0 per Kaggle. Do not commit raw CSVs.
