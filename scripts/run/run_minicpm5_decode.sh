#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

CUSTOM_OPP_VENDOR="${MINICPM5_CUSTOM_OPP_VENDOR:-${MINICPMV_CUSTOM_OPP_VENDOR:-$REPO_ROOT/custom_opp_install/vendors/customize}}"
if [ "${SKIP_CUSTOM_OPS:-0}" != "1" ] && [ ! -d "$CUSTOM_OPP_VENDOR/op_api/lib" ]; then
  echo "custom ops not installed at $CUSTOM_OPP_VENDOR; running scripts/install_custom_ops.sh" >&2
  scripts/install_custom_ops.sh
fi

source scripts/set_env.sh

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
