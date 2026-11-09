#!/usr/bin/env bash
set -euo pipefail
export ATS_KIND=gen_eval ATS_CONFIG=gen_eval/main
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# GPU is reserved for numerical forecasting training; LLM calls use the remote API.
export CUDA_VISIBLE_DEVICES="${ATS_TRAIN_GPU:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run "${PROJECT_ROOT}/scripts/gen_eval/run_complete.py"
