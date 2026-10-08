#!/usr/bin/env bash
# Track-B controlled degradation sweep on Physiome (drop/span/subsample/noise).
set -euo pipefail
export ATS_KIND="${ATS_KIND:-gen_eval}"; export ATS_LOG_NAME=degradation
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run -m agent4ts.baselines.run_degradation --config "${ATS_CONFIG}" --kind "${ATS_KIND}" "$@"
