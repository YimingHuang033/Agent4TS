#!/usr/bin/env bash
# Serial, resumable formal queue authorized by the owner on 2026-11-11.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
cd "$PROJECT_ROOT"
mkdir -p log/gen_eval
exec 9>log/gen_eval/queue_1111.lock
flock -n 9 || { echo '1111 supervisor already running'; exit 2; }
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
state=log/gen_eval/queue_1111.state
child=''
stop_queue() {
  trap - TERM INT
  echo "STOPPED $(date -Is)" > "$state"
  if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
  exit 143
}
trap stop_queue TERM INT
wait_gpu() {
  while true; do
    free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    compute=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ "$free" -ge 20000 && -z "$compute" ]]; then break; fi
    echo "WAITING_GPU0 $(date -Is) free_mib=$free" > "$state"
    sleep 30 & child=$!; wait "$child"; child=''
  done
}
stage() {
  local name="$1"; shift
  wait_gpu
  echo "RUNNING $name $(date -Is)" | tee "$state"
  setsid bash "$@" >> "log/gen_eval/queue_1111_${name}.log" 2>&1 & child=$!
  local rc=0
  wait "$child" || rc=$?
  child=''
  echo "FINISHED $name rc=$rc $(date -Is)" | tee -a log/gen_eval/queue_1111_events.log
  # MetaCaster returns 1 after completing a sweep containing failed cells.
  # Failures remain visible for explicit retry; subsequent phases still run.
  if [[ "$rc" -ne 0 && "$name" != meta_full && "$name" != meta_k30 ]]; then
    echo "ERROR $name rc=$rc $(date -Is)" > "$state"; exit "$rc"
  fi
}
printf '["6266c5d9543ac34eea365707"]\n' > log/gen_eval/retry_1111.json
stage matched_retry scripts/gen_eval/18_resume_legacy_1111.sh matched_space --retry "$PROJECT_ROOT/log/gen_eval/retry_1111.json"
stage matched_space scripts/gen_eval/18_resume_legacy_1111.sh matched_space
stage meta_full scripts/gen_eval/17_run_metacaster.sh
stage meta_k30 scripts/gen_eval/17_run_metacaster.sh --shots 30
stage budget scripts/gen_eval/18_resume_legacy_1111.sh budget
stage robustness scripts/gen_eval/18_resume_legacy_1111.sh robustness
stage initialization scripts/gen_eval/18_resume_legacy_1111.sh initialization
python_cmd="${HOME}/miniconda3/envs/tim/bin/python"
"$python_cmd" scripts/gen_eval/19_status_1111.py > log/gen_eval/queue_1111_final_status.json
# Completion means every sweep finished, not that every cell succeeded.
echo "SWEEPS_FINISHED_REVIEW_FAILURES $(date -Is)" > "$state"
