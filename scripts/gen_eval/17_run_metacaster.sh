#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
cd "$PROJECT_ROOT"
ats_run scripts/gen_eval/run_metacaster_1111.py "$@"
