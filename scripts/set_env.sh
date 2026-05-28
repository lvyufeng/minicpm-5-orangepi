#!/usr/bin/env bash
# Source this script before running any engine binary. Sets the Ascend toolkit
# paths and the custom_opp vendor path.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

ASCEND_ROOT="${MINICPM5_ASCEND_TOOLKIT_ROOT:-${MINICPMV_ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_TOOLKIT_ROOT="$ASCEND_ROOT/aarch64-linux"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_ROOT/lib64:${LD_LIBRARY_PATH:-}"

CUSTOM_OPP_VENDOR="${MINICPM5_CUSTOM_OPP_VENDOR:-${MINICPMV_CUSTOM_OPP_VENDOR:-$REPO_ROOT/custom_opp_install/vendors/customize}}"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
