#!/usr/bin/env bash
# Interpretability: analyse action traces and per-round evidence from a search run.
set -euo pipefail
export ATS_KIND=interp; export ATS_CONFIG="${ATS_CONFIG:-gen_eval/main}"; export ATS_LOG_NAME=trace_analysis
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run -m agent4ts.baselines.run_interp --config "${ATS_CONFIG}"
