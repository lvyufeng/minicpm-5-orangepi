#!/usr/bin/env bash
set -euo pipefail

WEIGHTS=${WEIGHTS:-models/MiniCPM5-1B}
INPUT_IDS=${INPUT_IDS:-0}
MAX_NEW=${MAX_NEW:-16}
MAX_SEQ=${MAX_SEQ:-4096}
BUILD_DIR=${BUILD_DIR:-build}

cmake -S . -B "$BUILD_DIR" -DMINICPM5_ENABLE_ENGINE=ON
cmake --build "$BUILD_DIR" --target minicpm5_decode -j"$(nproc)"
"$BUILD_DIR/minicpm5_decode" \
  --weights "$WEIGHTS" \
  --input-ids "$INPUT_IDS" \
  --max-new "$MAX_NEW" \
  --max-seq "$MAX_SEQ"
