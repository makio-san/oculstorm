#!/usr/bin/env bash
# Download the training data into the current worktree, or remove it again.
#
# Usage (from inside an agent worktree):
#   bash agents/bin/data.sh fetch   # data/kaggle/omni2_hourly.parquet from Kaggle
#   bash agents/bin/data.sh raw     # data/raw/omni2_all_years.dat from NASA SPDF (newest hours)
#   bash agents/bin/data.sh clean   # delete both; always run this before the run ends
#
# The disk is nearly full, so data lives only for the length of one run.
set -euo pipefail

KAGGLE=/Users/esma/Projects/oculstorm/.venv/bin/kaggle
DATASET=makiosann/omni2-hourly-dst
PARQUET=data/kaggle/omni2_hourly.parquet
RAW=data/raw/omni2_all_years.dat

[ -f oculstorm/data.py ] || { echo "run this from the root of an oculstorm worktree" >&2; exit 1; }

case "${1:-}" in
  fetch)
    if [ ! -f "$PARQUET" ]; then
      tmp=$(mktemp -d)
      "$KAGGLE" datasets download -d "$DATASET" -p "$tmp" --unzip -q
      mv "$tmp/omni2_hourly.parquet" "$PARQUET"
      rm -rf "$tmp"
    fi
    ls -lh "$PARQUET"
    ;;
  raw)
    mkdir -p data/raw
    curl -fsS -o "$RAW" https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat
    ls -lh "$RAW"
    ;;
  clean)
    rm -f "$PARQUET" "$RAW"
    echo "data removed"
    ;;
  *)
    echo "usage: data.sh fetch|raw|clean" >&2
    exit 2
    ;;
esac
