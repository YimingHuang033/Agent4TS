#!/usr/bin/env bash
# Train the fixed (non-agent) models and write results/<kind>/benchmark.{json,csv}.
set -euo pipefail
export ATS_KIND=gen_eval; export ATS_LOG_NAME=benchmark
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" benchmark "$@"
