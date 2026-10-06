#!/usr/bin/env bash
# Serve Qwen3.5-4B with vLLM on the local port used by config/default.yaml.
# Used by every experiment that needs the LLM controller. Keep it running in the
# background; scripts/…/_env.sh points ATS_LLM_BASE_URL at this endpoint.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ATS_KIND=gen_eval

MODEL_DIR="${ATS_DATA_ROOT}/../Qwen3.5-4B"          # /mnt/data/Qwen3.5-4B
HOST="127.0.0.1"; PORT=8848
LOG_FILE="${LOG_DIR}/vllm_serve.log"

echo "== serving ${MODEL_DIR} on ${HOST}:${PORT}" | tee -a "${LOG_FILE}"
conda run -n "${ATS_ENV}" --no-capture-output \
  vllm serve "${MODEL_DIR}" \
    --host "${HOST}" --port "${PORT}" \
    --served-model-name Qwen3.5-4B \
    --gpu-memory-utilization 0.45 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --max-num-seqs 64 \
  >>"${LOG_FILE}" 2>&1
