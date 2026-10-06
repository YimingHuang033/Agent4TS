#!/usr/bin/env bash
# Agent search (needs the LLM served by 01_serve_llm.sh).
#   scripts/gen_eval/05_search.sh [--tasks a,b] [--seeds 0,1] [--ablation no_vision]
set -euo pipefail
export ATS_KIND=gen_eval; export ATS_LOG_NAME=search
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
ats_run_cli --config "${ATS_CONFIG}" --kind "${ATS_KIND}" search "$@"
