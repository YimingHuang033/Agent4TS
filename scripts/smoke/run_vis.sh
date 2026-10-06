#!/usr/bin/env bash
# Render result figures + HTML index for a results kind (default: smoke).
set -euo pipefail
export ATS_KIND="${ATS_KIND:-smoke}"; export ATS_LOG_NAME=vis
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run "${PROJECT_ROOT}/vis/plot_results.py" --kind "${1:-smoke}"
