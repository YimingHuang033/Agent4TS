#!/usr/bin/env bash
# Performance / cost eval: wall-clock, peak memory, tokens per trial.
set -euo pipefail
export ATS_KIND=perf; export ATS_CONFIG="${ATS_CONFIG:-gen_eval/main}"; export ATS_LOG_NAME=perf_benchmark
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run -m agent4ts.baselines.run_perf --config "${ATS_CONFIG}"
