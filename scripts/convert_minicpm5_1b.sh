#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models/MiniCPM5-1B}
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/minicpm5-1b}

python3 scripts/convert/convert_minicpm5_1b.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR"
