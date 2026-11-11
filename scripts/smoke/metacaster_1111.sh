#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export ATS_KIND=smoke
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
cd "$PROJECT_ROOT"
ats_run scripts/gen_eval/run_metacaster_1111.py --smoke "$@"
