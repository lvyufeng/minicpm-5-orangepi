#!/usr/bin/env bash
# Build the full engine into $REPO_ROOT/build (or $BUILD_DIR if set).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

cd "$REPO_ROOT"
scripts/install_custom_ops.sh
source scripts/set_env.sh

mkdir -p "$BUILD_DIR"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DMINICPM5_ENABLE_ENGINE=ON
cmake --build "$BUILD_DIR" -j"$(nproc)"
