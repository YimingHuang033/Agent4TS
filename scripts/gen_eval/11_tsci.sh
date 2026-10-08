#!/usr/bin/env bash
# TSci-adapted baseline. Reports its true availability; refuses to fake TSci.
set -euo pipefail
export ATS_KIND="${ATS_KIND:-gen_eval}"; export ATS_LOG_NAME=tsci
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run -m agent4ts.baselines.run_tsci --config "${ATS_CONFIG}"
