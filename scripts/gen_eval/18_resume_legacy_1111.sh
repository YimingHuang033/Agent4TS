#!/usr/bin/env bash
# Invoke only after the project owner authorizes resuming the stopped v3 queue.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
cd "$PROJECT_ROOT"
if [[ $# -eq 0 ]]; then
  echo 'Usage: bash scripts/gen_eval/18_resume_legacy_1111.sh <phase> [--retry JSON] [--limit N]'
  exit 2
fi
legacy_dir="${PROJECT_ROOT}_v3_frozen_1111"
if [[ ! -e "$legacy_dir" ]]; then
  mkdir "$legacy_dir"
  git archive a8f19bd648d2d95cdd9c928e0b55d1939327667e | tar -x -C "$legacy_dir"
  tar -xzf "$PROJECT_ROOT/results/gen_eval/v3_expand_20261109/source_snapshot.tar.gz" -C "$legacy_dir"
  # git archive contains tracked smoke output; leave it intact, link only gen_eval.
  mkdir -p "$legacy_dir/results"
  ln -s "$PROJECT_ROOT/results/gen_eval" "$legacy_dir/results/gen_eval"
fi
# Explicit v3 source prevents new modules changing its cache/source identity.
export PYTHONPATH="$legacy_dir/src"
cd "$legacy_dir"
exec bash scripts/gen_eval/15_run_expand.sh "$@"
