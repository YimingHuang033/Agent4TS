#!/usr/bin/env bash
# Real-LLM smoke: run one search round against the locally served Qwen3.5-4B.
# Requires scripts/gen_eval/01_serve_llm.sh to be running (127.0.0.1:8848).
set -euo pipefail
# the client side runs CPU-only; only the separately launched vLLM server uses GPU
export CUDA_VISIBLE_DEVICES=""
export ATS_KIND=smoke
export ATS_CONFIG=smoke/llm
export ATS_LOG_NAME=llm_smoke
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

echo "== waiting for LLM endpoint" | tee -a "${LOG_FILE}"
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:8848/v1/models" >/dev/null 2>&1; then break; fi
    sleep 5
    [ "$i" = 60 ] && { echo "LLM endpoint never came up" | tee -a "${LOG_FILE}"; exit 1; }
done
curl -s "http://127.0.0.1:8848/v1/models" | tee -a "${LOG_FILE}"; echo

ats_run_cli --config "${ATS_CONFIG}" --kind smoke search --tasks physiome_dupont --seeds 0
echo "== llm smoke done" | tee -a "${LOG_FILE}"
