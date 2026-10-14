#!/usr/bin/env bash
# TSRouter baseline (dynamic modality-model routing). Reports its true
# availability; refuses to fake TSRouter.
set -euo pipefail
export ATS_KIND="${ATS_KIND:-gen_eval}"; export ATS_LOG_NAME=tsrouter
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run -m agent4ts.baselines.run_tsrouter --config "${ATS_CONFIG}"
