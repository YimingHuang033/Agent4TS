#!/usr/bin/env bash
set -euo pipefail
export ATS_KIND=gen_eval
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "$0")/../_env.sh"
cd "$PROJECT_ROOT"
ats_run scripts/gen_eval/run_expand.py "$@"
