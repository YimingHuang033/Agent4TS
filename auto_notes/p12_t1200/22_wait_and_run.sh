#!/usr/bin/env bash
# Wait for the 1111 queue to finish, then run the p12 timeout-1200 rerun (C plan).
# Detached, idempotent, cancellable:  kill "$(cat log/gen_eval/p12_t1200_waiter.pid)"
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/tim/Proj/ats
PIDFILE="$PROJECT_ROOT/log/gen_eval/queue_1111.pid"
LOCK=/tmp/ats-p12t1200-waiter.lock
LOG="$PROJECT_ROOT/log/gen_eval/p12_t1200_waiter.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

# single instance
exec 9>"$LOCK"
if ! flock -n 9; then
  log "another waiter already holds $LOCK; exiting"
  exit 0
fi
echo $$ >"$PROJECT_ROOT/log/gen_eval/p12_t1200_waiter.pid"
log "waiter started (pid $$); will launch the p12 timeout-1200 rerun once queue 1111 is gone"

queue_alive() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then return 0; fi
  if pgrep -f 'scripts/gen_eval/20_launch_1111.sh' >/dev/null 2>&1; then return 0; fi
  return 1
}

gpu_busy() {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [[ "$n" -gt 0 ]]
}

ticks=0
while queue_alive || gpu_busy; do
  if (( ticks % 10 == 0 )); then
    state=$(cat "$PROJECT_ROOT/log/gen_eval/queue_1111.state" 2>/dev/null || echo 'unknown')
    log "waiting: queue_alive=$(queue_alive && echo yes || echo no) gpu_busy=$(gpu_busy && echo yes || echo no) state='$state'"
  fi
  ticks=$((ticks+1))
  sleep 60
done

log "queue is finished and GPU is idle; starting rerun"
cd "$HERE"
bash 21_rerun_p12_t1200.sh run >>"$LOG" 2>&1
rc=$?
log "rerun exited rc=$rc"
exit "$rc"
