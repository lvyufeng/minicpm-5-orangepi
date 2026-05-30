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

BUILD_DIR=${BUILD_DIR:-build}
WEIGHTS=${WEIGHTS:-models/MiniCPM5-1B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7860}
MAX_SEQ=${MAX_SEQ:-4096}
MAX_NEW=${MAX_NEW:-256}
TOKENIZER=${TOKENIZER:-$WEIGHTS}

cmake -S . -B "$BUILD_DIR" -DMINICPM5_ENABLE_ENGINE=ON
cmake --build "$BUILD_DIR" --target minicpm5_server -j"$(nproc)"

export BUILD_DIR WEIGHTS HOST PORT MAX_SEQ MAX_NEW TOKENIZER
export USE_TORCH=${USE_TORCH:-0}
export MINICPM5_SERVER_BIN="$REPO_ROOT/$BUILD_DIR/minicpm5_server"

python3 -m uvicorn app:app --host "$HOST" --port "$PORT"
