#!/usr/bin/env bash
# Smoke test: mock LLM, tiny budget, no GPU-heavy models. Proves the whole loop
# (load -> validate -> trial -> feedback -> archive -> export) runs end to end.
set -euo pipefail
# debugging/smoke never occupies a GPU; vLLM serving is a separate, explicit step
export CUDA_VISIBLE_DEVICES=""
export ATS_KIND=smoke
export ATS_CONFIG=smoke/quick
export ATS_LOG_NAME=smoke_test
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

ats_run_cli --config "${ATS_CONFIG}" --kind smoke prepare
ats_run_cli --config "${ATS_CONFIG}" --kind smoke validate
ats_run_cli --config "${ATS_CONFIG}" --kind smoke benchmark --models LastValue,MeanValue,DLinear
ats_run_cli --config "${ATS_CONFIG}" --kind smoke search --tasks physiome_dupont --seeds 0
ats_run_cli --config "${ATS_CONFIG}" --kind smoke evaluate --tasks physiome_dupont --seed 0
echo "== smoke done; results in results/smoke" | tee -a "${LOG_FILE}"
