#!/usr/bin/env bash
# Prepare: load every configured task, report which are ok vs blocked.
set -euo pipefail
export ATS_KIND=gen_eval; export ATS_LOG_NAME=prepare
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" prepare
