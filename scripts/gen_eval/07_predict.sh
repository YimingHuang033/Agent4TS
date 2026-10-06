#!/usr/bin/env bash
# Offline prediction from a frozen pipeline (no LLM):
#   scripts/gen_eval/07_predict.sh --pipeline results/gen_eval/search/<task>/seed0/best_pipeline.json --tasks <task>
set -euo pipefail
export ATS_KIND=gen_eval; export ATS_LOG_NAME=predict
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" predict "$@"
