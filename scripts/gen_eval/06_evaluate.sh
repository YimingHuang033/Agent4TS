#!/usr/bin/env bash
# Deterministic confirm+test on locked candidates (no LLM feedback afterwards).
set -euo pipefail
export ATS_KIND=gen_eval; export ATS_LOG_NAME=evaluate
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" evaluate "$@"
