#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE=${MODEL_FILE:-models/MiniCPM5-1B}
BUILD_DIR=${BUILD_DIR:-build}

cmake -S . -B "$BUILD_DIR" -DMINICPM5_ENABLE_ENGINE=OFF
cmake --build "$BUILD_DIR" --target minicpm5_validate_weights -j"$(nproc)"
"$BUILD_DIR/minicpm5_validate_weights" "$MODEL_FILE"
