#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export ATS_KIND=smoke ATS_CONFIG=smoke/ark ATS_LOG_NAME=llm_smoke
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind smoke --run-dir "${RESULT_DIR}/ark_v2" search --tasks physiome_dupont --seeds 0 "$@"
