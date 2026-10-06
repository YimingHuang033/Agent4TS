#!/usr/bin/env bash
# Random-search baseline over the same action space and budget.
set -euo pipefail
export ATS_KIND=gen_eval; export ATS_LOG_NAME=random_search
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run -m agent4ts.baselines.run_random --config "${ATS_CONFIG}" --kind "${ATS_KIND}" "$@"
