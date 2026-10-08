#!/usr/bin/env bash
# Deterministic data checks: no leakage, split isolation, padding invariance.
set -euo pipefail
export ATS_KIND="${ATS_KIND:-gen_eval}"; export ATS_LOG_NAME=validate
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" validate
