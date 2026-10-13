#!/usr/bin/env bash
# Download the competition data once ~/.kaggle/kaggle.json is in place.
set -euo pipefail

COMP="x-hec-ts-2025-predicting-air-quality-in-paris"
RAW="$(cd "$(dirname "$0")" && pwd)/data/raw"

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
  echo "No Kaggle credentials found at ~/.kaggle/kaggle.json" >&2
  echo "Create one at https://www.kaggle.com/settings -> Create New Token" >&2
  exit 1
fi
chmod 600 "$HOME/.kaggle/"* 2>/dev/null || true

mkdir -p "$RAW"
python3 -m kaggle competitions download -c "$COMP" -p "$RAW"
unzip -o "$RAW/$COMP.zip" -d "$RAW"
rm -f "$RAW/$COMP.zip"
ls -la "$RAW"
