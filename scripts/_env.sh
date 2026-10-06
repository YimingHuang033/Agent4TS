#!/usr/bin/env bash
# Shared environment for every Agent4TS script.
# Source this; do not execute it directly.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export ATS_ENV="${ATS_ENV:-tim}"
export ATS_KIND="${ATS_KIND:-gen_eval}"
export ATS_CONFIG="${ATS_CONFIG:-default}"

# data/resources live outside the repo and never enter git
export ATS_DATA_ROOT="${ATS_DATA_ROOT:-/mnt/data/ats_resources}"

# mirror endpoints (see changyong.txt)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GIT_MIRROR="${GIT_MIRROR:-https://ghproxy.net/}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LOG_DIR="${PROJECT_ROOT}/log/${ATS_KIND}"
RESULT_DIR="${PROJECT_ROOT}/results/${ATS_KIND}"
CONFIG_DIR="${PROJECT_ROOT}/config"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${CONFIG_DIR}/${ATS_KIND}"

LOG_FILE="${LOG_DIR}/${ATS_LOG_NAME:-$(basename "${BASH_SOURCE[1]:-run}" .sh)}.log"

# run a command inside the conda env, echoing it for the log
ats_run() {
    echo "== [$(date '+%F %T')] conda run -n ${ATS_ENV} python $*" | tee -a "${LOG_FILE}"
    conda run -n "${ATS_ENV}" --no-capture-output python "$@" 2>&1 | tee -a "${LOG_FILE}"
}

ats_run_cli() {
    ats_run -m agent4ts.cli "$@"
}
