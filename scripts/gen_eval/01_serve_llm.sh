#!/usr/bin/env bash
# Serve Qwen3.5-4B with vLLM on the local port used by config/default.yaml.
# Used by every experiment that needs the LLM controller. Keep it running in the
# background; scripts/…/_env.sh points ATS_LLM_BASE_URL at this endpoint.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ATS_KIND=gen_eval

# Overridable so a smaller model can validate the loop while the GPUs are busy:
#   ATS_VLLM_MODEL_DIR=/mnt/data/Qwen2.5-0.5B-Instruct ATS_VLLM_UTIL=0.15 \
#   ATS_VLLM_NAME=Qwen2.5-0.5B ATS_VLLM_EXTRA="--enforce-eager" scripts/gen_eval/01_serve_llm.sh
MODEL_DIR="${ATS_VLLM_MODEL_DIR:-/mnt/data/Qwen3.5-4B}"
SERVED_NAME="${ATS_VLLM_NAME:-Qwen3.5-4B}"
UTIL="${ATS_VLLM_UTIL:-0.85}"
MAXLEN="${ATS_VLLM_MAXLEN:-8192}"
HOST="127.0.0.1"; PORT="${ATS_VLLM_PORT:-8848}"
LOG_FILE="${LOG_DIR}/vllm_serve.log"

echo "== serving ${MODEL_DIR} as ${SERVED_NAME} on ${HOST}:${PORT} (util=${UTIL})" | tee -a "${LOG_FILE}"
# find the target env's bin dir so we can exec vllm directly (conda run drops
# CUDA_HOME/PATH, which FlashInfer JIT needs on this machine)
ENV_PREFIX="$(conda run -n "${ATS_ENV}" python -c 'import sys;print(sys.prefix)')"
export PATH="${ENV_PREFIX}/bin:${NVCC_BIN:-}:${PATH}"
"${ENV_PREFIX}/bin/vllm" serve "${MODEL_DIR}" \
    --host "${HOST}" --port "${PORT}" \
    --served-model-name "${SERVED_NAME}" \
    --gpu-memory-utilization "${UTIL}" \
    --max-model-len "${MAXLEN}" \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    ${ATS_VLLM_EXTRA:-} \
  >>"${LOG_FILE}" 2>&1
