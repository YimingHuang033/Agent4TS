#!/usr/bin/env bash
# Operational supervisor; does not change the frozen experiment code/config.
set -euo pipefail
export ATS_KIND=gen_eval
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
source "$(dirname "$0")/../_env.sh"
cd "$PROJECT_ROOT"
RUN_ROOT="$PROJECT_ROOT/results/gen_eval/v3_expand_20261109"
STATE="$RUN_ROOT/launch_state.txt"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/launch.lock"
flock -n 9 || { echo 'Another expansion supervisor holds launch.lock'; exit 73; }
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"
phase=preflight
state() { printf '%s phase=%s %s\n' "$(date -Is)" "$phase" "$*" | tee "$STATE"; }
on_exit() { code=$?; if (( code != 0 )); then state "FAILED exit=$code; inspect supervisor log; no automatic retry"; fi; }
trap on_exit EXIT
state 'PREFLIGHT verifying frozen manifest'
CUDA_VISIBLE_DEVICES="" ats_run scripts/gen_eval/run_expand.py manifest
for phase in new_data matched_space budget data_ablation robustness initialization; do
    while true; do
        free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= 20000 && processes == 0 )); then break; fi
        state "WAITING_GPU gpu=0 free_mib=$free compute_processes=$processes; require >=20000 MiB and no compute process"
        sleep 60
    done
    state 'RUNNING numerical models on GPU 0; LLM via Ark API only'
    CUDA_VISIBLE_DEVICES=0 bash scripts/gen_eval/15_run_expand.sh "$phase"
    state 'PHASE_FINISHED; individual failures remain recorded, not silently retried'
    CUDA_VISIBLE_DEVICES="" ats_run scripts/gen_eval/status_expand.py
    CUDA_VISIBLE_DEVICES="" ats_run scripts/gen_eval/summarize_expand.py
 done
phase=all
state 'EXECUTION_QUEUE_FINISHED; inspect statuses and failures before scientific acceptance'
