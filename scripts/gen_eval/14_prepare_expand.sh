#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export ATS_KIND=smoke
source "$(dirname "$0")/../_env.sh"
cd "$PROJECT_ROOT"
ats_run scripts/gen_eval/download_expand.py
ats_run scripts/gen_eval/prepare_expand.py
