#!/usr/bin/env bash
# Monitor the p12 timeout-1200 rerun until all 6 cells reach a final status.
# Writes one status line every 5 minutes to log/gen_eval/p12_t1200_monitor.log and
# prints a final comparison against the old 600s root when done.
set -uo pipefail

PROJECT_ROOT=/home/tim/Proj/ats
NEW_ROOT="$PROJECT_ROOT/results/gen_eval/v3_p12t1200_20261111"
OLD_ROOT="$PROJECT_ROOT/results/gen_eval/v3_expand_20261109"
LOG="$PROJECT_ROOT/log/gen_eval/p12_t1200_monitor.log"
PY=/home/tim/miniconda3/envs/tim/bin/python
UNIT=p12-t1200-1111
EXPECT=6

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

status_line() {
  "$PY" - "$NEW_ROOT" <<'PY'
import json, glob, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for d in sorted((root/'cells').glob('*')) if (root/'cells').exists() else []:
    attempts = sorted(glob.glob(str(d/'attempt*')))
    final = None
    for a in attempts:
        f = pathlib.Path(a)/'result.json'
        if f.exists():
            try: final = json.loads(f.read_text())
            except Exception: final = {'status': 'unparsable'}
    if final is None: continue
    rows.append((final.get('task'), final.get('seed'), final.get('status'),
                 round(float(final.get('wall_s') or 0)), final.get('attempt'), final.get('error') or ''))
for t, s, st, w, a, e in sorted(rows):
    print(f'  {t} seed{s} attempt{a} {st} wall={w}s {e[:80]}')
done = [r for r in rows if r[2] in ('ok', 'error')]
print(f'  cells with final status: {len(done)}/{6}')
PY
}

compare() {
  "$PY" - "$NEW_ROOT" "$OLD_ROOT" <<'PY'
import json, glob, pathlib, sys
new_root, old_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def collect(root):
    out = {}
    for d in sorted((root/'cells').glob('*')):
        attempts = sorted(glob.glob(str(d/'attempt*')))
        final = None
        for a in attempts:
            f = pathlib.Path(a)/'result.json'
            if f.exists():
                try: final = json.loads(f.read_text())
                except Exception: pass
        if final and final.get('method') == 'tsci_model_only' and str(final.get('task','')).startswith('p12'):
            out[(final['task'], final['seed'])] = (final.get('status'), round(float(final.get('wall_s') or 0)), final.get('error') or '')
    return out
new, old = collect(new_root), collect(old_root)
print(f"{'task':<14}{'seed':<6}{'1200s(new)':<28}{'600s(old)'}")
for k in sorted(new):
    ns, nw, ne = new[k]
    os_, ow, oe = old.get(k, ('missing', 0, 'no attempt in old root'))
    print(f"{k[0]:<14}{k[1]:<6}{ns+f' ({nw}s)':<28}{os_} ({ow}s)")
PY
}

log "monitor started"
while true; do
  log "---- $(systemctl --user is-active "$UNIT" 2>/dev/null) | $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n '2p') | queue: $(cat "$PROJECT_ROOT/log/gen_eval/queue_1111.state" 2>/dev/null)"
  status_line >>"$LOG"
  done_count=$("$PY" - "$NEW_ROOT" <<'PY'
import json, glob, pathlib, sys
root = pathlib.Path(sys.argv[1]); n = 0
for d in sorted((root/'cells').glob('*')) if (root/'cells').exists() else []:
    attempts = sorted(glob.glob(str(d/'attempt*')))
    final = None
    for a in attempts:
        f = pathlib.Path(a)/'result.json'
        if f.exists():
            try: final = json.loads(f.read_text())
            except Exception: pass
    if final and final.get('status') in ('ok', 'error'): n += 1
print(n)
PY
)
  active=$(systemctl --user is-active "$UNIT" 2>/dev/null || true)
  if [[ "$done_count" -ge "$EXPECT" ]]; then log "all cells final ($done_count/$EXPECT)"; break; fi
  if [[ "$active" != 'active' && "$active" != 'activating' ]]; then
    log "unit is no longer active ($active) with $done_count/$EXPECT cells final"; break
  fi
  sleep 300
done

log "==== FINAL COMPARISON (new 1200s vs old 600s) ===="
compare >>"$LOG"
log "monitor finished"
cat "$LOG" | tail -40
