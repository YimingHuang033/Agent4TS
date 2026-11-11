#!/usr/bin/env bash
# Post-queue follow-up for the 1111 v3 sweep.
#
# Purpose: rerun the p12 tsci_model_only matched_space cells with
# trial_timeout_seconds=1200, in an isolated new root, because 600s is marginal
# for PatchTST on p12 (cell 6266c5d9543ac34eea365707: 11/11 candidates timed out
# in attempt002, 4/10 succeeded in attempt001 on the same grid).
#
# Hard rules honoured here:
#   * never writes to /home/tim/Proj/ats_v3_frozen_1111 (frozen hashes stay valid)
#   * never writes to results/gen_eval/v3_expand_20261109 (completed cells untouched)
#   * never writes to src/ config/ scripts/gen_eval/ of the main repo (that would
#     change the MetaCaster manifest identity and break meta_full / meta_k30)
#   * refuses to run while the 1111 queue is alive
#
# Usage:
#   bash .../21_rerun_p12_t1200.sh plan   # guard + build copy + manifest + list cells (no GPU)
#   bash .../21_rerun_p12_t1200.sh run    # the same, then actually train on GPU0
set -euo pipefail

MODE="${1:-plan}"
PROJECT_ROOT=/home/tim/Proj/ats
SRC_FROZEN="${PROJECT_ROOT}_v3_frozen_1111"
NEWDST="${PROJECT_ROOT}_v3_p12t1200_1111"
NEW_ROOT_REL="results/gen_eval/v3_p12t1200_20261111"
ORIG_ROOT_REL="results/gen_eval/v3_expand_20261109"
BASE_COMMIT=a8f19bd648d2d95cdd9c928e0b55d1939327667e
FILTER=p12_native,p12_robust
PY=/home/tim/miniconda3/envs/tim/bin/python

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------- 1. guard: the 1111 queue must be finished ----------
pidfile="$PROJECT_ROOT/log/gen_eval/queue_1111.pid"
queue_alive=0
if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then queue_alive=1; fi
if pgrep -f 'scripts/gen_eval/20_launch_1111.sh' >/dev/null 2>&1; then queue_alive=1; fi
if [[ "$queue_alive" -eq 1 ]]; then
  if [[ "$MODE" == 'plan' && "${ALLOW_PLAN_WHILE_QUEUED:-0}" == '1' ]]; then
    log "WARNING: queue 1111 is alive. plan continues only because ALLOW_PLAN_WHILE_QUEUED=1;"
    log "         this path is offline (CUDA_VISIBLE_DEVICES='') and writes only into $NEWDST and $NEW_ROOT_REL."
  else
    die "queue 1111 still running (pid $(cat "$pidfile" 2>/dev/null)). Wait for SWEEPS_FINISHED_REVIEW_FAILURES, then re-run."
  fi
fi

# ---------- 2. isolate: copy from the same commit + frozen snapshot ----------
if [[ ! -d "$NEWDST" ]]; then
  log "creating $NEWDST from $BASE_COMMIT + source_snapshot.tar.gz"
  mkdir -p "$NEWDST"
  git -C "$PROJECT_ROOT" archive "$BASE_COMMIT" | tar -x -C "$NEWDST"
  tar -xzf "$PROJECT_ROOT/$ORIG_ROOT_REL/source_snapshot.tar.gz" -C "$NEWDST"
  mkdir -p "$NEWDST/results"
  ln -sfn "$PROJECT_ROOT/results/gen_eval" "$NEWDST/results/gen_eval"
  [[ -f "$NEWDST/results/smoke/expand_1109/debug_passed.json" ]] \
    || ln -sfn "$PROJECT_ROOT/results/smoke" "$NEWDST/results/smoke"
else
  log "reusing existing $NEWDST"
fi

# ---------- 3. patch the copy only (timeout, new root, task filter) ----------
"$PY" - "$NEWDST" "$NEW_ROOT_REL" "$ORIG_ROOT_REL" <<'PY'
import pathlib, re, sys
dst, new_root, orig_root = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert new_root != orig_root
cfg = dst / 'config/gen_eval/expand_v3.yaml'
text = cfg.read_text()
text = text.replace(f'root: {orig_root}', f'root: {new_root}')
text = re.sub(r'trial_timeout_seconds:\s*600', 'trial_timeout_seconds: 1200', text)
assert f'root: {new_root}' in text and 'trial_timeout_seconds: 1200' in text, 'config patch failed'
cfg.write_text(text)

runner = dst / 'scripts/gen_eval/run_expand.py'
src = runner.read_text()
if "keep=os.environ.get('ATS_TASK_FILTER')" not in src:
    anchor = ("    for name in cfg['expansion']['new_tasks']:"
              "cfg['tasks'][name]={'backend':name.removesuffix('_forecast')}\n")
    assert anchor in src, 'anchor for task filter not found; run_expand.py changed upstream'
    inject = (anchor +
              "    keep=os.environ.get('ATS_TASK_FILTER')\n"
              "    if keep:\n"
              "        keep=set(keep.split(','))\n"
              "        cfg['tasks']={k:v for k,v in cfg['tasks'].items() if k in keep}\n"
              "        cfg['expansion']['new_tasks']=[t for t in cfg['expansion']['new_tasks'] if t in keep]\n")
    src = src.replace(anchor, inject, 1)
    src = src.replace('import argparse,copy,', 'import argparse,copy,os,', 1)
    assert 'import argparse,copy,os,' in src, 'import patch failed'
if 'if task not in bundles:return' not in src:
    # generate() references hard-coded task names (new_data / initialization); with a task
    # filter those tasks are not loaded, so unknown tasks must be skipped instead of KeyError.
    # Insert the guard as add()'s first statement: version-agnostic, matches any body layout.
    add_def = "    def add(phase,task,method,budget=12,f0='APN',degradation=None):\n"
    assert src.count(add_def) == 1, 'add() definition not unique; run_expand.py changed upstream'
    src = src.replace(add_def, add_def + "        if task not in bundles:return\n", 1)
runner.write_text(src)
print('patched config + runner in copy')
PY

# the frozen originals must stay byte-identical
"$PY" - "$SRC_FROZEN" <<'PY'
import hashlib, pathlib, sys, json
d = pathlib.Path(sys.argv[1])
h = {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
     for p in (d/'config/gen_eval/expand_v3.yaml', d/'scripts/gen_eval/run_expand.py')}
print('frozen originals sha256:', json.dumps(h, indent=1))
PY

# ---------- 4. freeze the new manifest offline (no GPU, p12 tasks only) ----------
cd "$NEWDST"
export ATS_TASK_FILTER="$FILTER"
export CUDA_VISIBLE_DEVICES=''
export ATS_LOG_NAME=21_manifest_p12_t1200.log
# shellcheck disable=SC1091
source scripts/_env.sh
ats_run scripts/gen_eval/run_expand.py manifest

# ---------- 5. select the p12 tsci cells to rerun ----------
RETRY_JSON="$PROJECT_ROOT/log/gen_eval/retry_p12_t1200.json"
"$PY" - "$NEWDST/$NEW_ROOT_REL/manifest.json" "$RETRY_JSON" <<'PY'
import json, pathlib, sys
man = json.loads(pathlib.Path(sys.argv[1]).read_text())
ids = [c['id'] for c in man['cells']
       if c['identity']['method'] == 'tsci_model_only'
       and c['identity']['task'].startswith('p12')
       and 'matched_space' in c['phases']]
pathlib.Path(sys.argv[2]).write_text(json.dumps(ids, indent=1))
print(f'manifest cells: {len(man["cells"])}; selected tsci p12 cells: {len(ids)}')
for c in man['cells']:
    i = c['identity']
    if i['method'] == 'tsci_model_only' and i['task'].startswith('p12'):
        print(' ', c['id'], i['task'], 'seed', i['seed'])
PY

if [[ "$MODE" != 'run' ]]; then
  log "plan complete. To execute:  bash $0 run"
  exit 0
fi

# ---------- 6. run, serially, GPU0 only, after checking it is idle ----------
# re-check: train never starts while the 1111 queue is alive, whatever the mode flag said
if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then
  die "queue 1111 became active again; refusing to train in parallel."
fi
if pgrep -f 'scripts/gen_eval/20_launch_1111.sh' >/dev/null 2>&1; then
  die "20_launch_1111.sh is in the process table; refusing to train in parallel."
fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[[ "$busy" -eq 0 ]] || die "GPU still has $busy compute process(es); refusing to share the card."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log "launching p12 tsci rerun (timeout 1200s) -> $NEW_ROOT_REL"
exec bash scripts/gen_eval/15_run_expand.sh matched_space --retry "$RETRY_JSON"
