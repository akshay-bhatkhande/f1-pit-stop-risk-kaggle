#!/usr/bin/env bash
# Download Playground Series S6E5 (Predicting F1 Pit Stops) into data/raw/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"
SOURCE_FILE="${RAW}/DATA_SOURCE.txt"

warn_if_synthetic() {
  if [[ -f "${RAW}/train.csv" ]]; then
    rows="$(wc -l < "${RAW}/train.csv" | tr -d ' ')"
    # header + ~439k → ~439141 lines; demo is ~22k
    if [[ -f "${SOURCE_FILE}" ]] && grep -q "synthetic_demo" "${SOURCE_FILE}"; then
      echo "WARN: data/raw currently holds SYNTHETIC DEMO CSVs (not official Kaggle)."
      echo "      Metrics/models trained on this are demo-only. Continuing download will replace them."
      return 0
    fi
    if [[ "${rows}" -lt 100000 ]]; then
      echo "WARN: train.csv has ${rows} lines (official S6E5 train is ~439k rows)."
      echo "      Likely synthetic demo — official download will overwrite data/raw/*.csv."
    fi
  fi
}

if [[ -f "${RAW}/train.csv" && -f "${RAW}/test.csv" ]]; then
  if [[ -f "${SOURCE_FILE}" ]] && grep -q "kaggle_playground_s6e5" "${SOURCE_FILE}"; then
    echo "OK: official-marked Kaggle CSVs already present in data/raw/"
    exit 0
  fi
  warn_if_synthetic
  rows="$(wc -l < "${RAW}/train.csv" | tr -d ' ')"
  if [[ "${rows}" -ge 300000 ]]; then
    printf 'kaggle_playground_s6e5\nrows≈%s (heuristic; marker written by download script)\n' "${rows}" > "${SOURCE_FILE}"
    echo "OK: large train.csv present — marked as kaggle_playground_s6e5"
    exit 0
  fi
  echo "INFO: existing CSVs look like demo scale. Re-run with FORCE=1 to download official data."
  echo "      FORCE=1 bash scripts/download_data.sh"
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "OK: leaving existing files in place (FORCE=1 to overwrite with Kaggle)."
    exit 0
  fi
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
  echo "Owner path after token: FORCE=1 bash scripts/download_data.sh && python -m src.models.train"
  exit 1
fi

warn_if_synthetic
echo "Downloading playground-series-s6e5 ..."
kaggle competitions download -c playground-series-s6e5 -p "${RAW}"
ZIP="${RAW}/playground-series-s6e5.zip"
if [[ -f "${ZIP}" ]]; then
  unzip -o "${ZIP}" -d "${RAW}"
  rm -f "${ZIP}"
fi

if [[ -f "${RAW}/train.csv" && -f "${RAW}/test.csv" ]]; then
  printf 'kaggle_playground_s6e5\nsource=kaggle competitions download -c playground-series-s6e5\n' > "${SOURCE_FILE}"
  echo "OK: official Kaggle data ready in data/raw/ (DATA_SOURCE.txt=kaggle_playground_s6e5)"
  echo "Next: python -m src.models.train   # retrain before trusting metrics / UI"
else
  echo "ERROR: download finished but CSVs missing. Check competition rule acceptance."
  exit 1
fi
