#!/usr/bin/env bash
# Download Playground Series S6E5 (Predicting F1 Pit Stops) into data/raw/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"

if [[ -f "${RAW}/train.csv" && -f "${RAW}/test.csv" ]]; then
  echo "OK: train.csv and test.csv already present in data/raw/"
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found."
  echo "  pip install kaggle"
  echo "  Or: python -m src.data.generate_demo   # offline synthetic fallback"
  echo "  Or: manual download → see data/raw/README.md"
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "ERROR: missing ~/.kaggle/kaggle.json"
  echo "  1. Create a free Kaggle account"
  echo "  2. Accept rules: https://www.kaggle.com/competitions/playground-series-s6e5/rules"
  echo "  3. Account → Settings → API → Create New Token → save as ~/.kaggle/kaggle.json"
  echo "  4. chmod 600 ~/.kaggle/kaggle.json"
  echo "Fallback: python -m src.data.generate_demo"
  exit 1
fi

echo "Downloading playground-series-s6e5 ..."
kaggle competitions download -c playground-series-s6e5 -p "${RAW}"
ZIP="${RAW}/playground-series-s6e5.zip"
if [[ -f "${ZIP}" ]]; then
  unzip -o "${ZIP}" -d "${RAW}"
  rm -f "${ZIP}"
fi

if [[ -f "${RAW}/train.csv" && -f "${RAW}/test.csv" ]]; then
  echo "OK: data ready in data/raw/"
else
  echo "ERROR: download finished but CSVs missing. Check competition rule acceptance."
  exit 1
fi
