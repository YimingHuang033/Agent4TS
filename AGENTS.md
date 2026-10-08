# AGENTS.md — working rules for this repository

Agent4TS: single-agent search over robust irregular time-series forecasting
pipelines. Spec in `DESIGN.md`; user-facing docs in `README.md`. Keep all three
documents consistent when behavior changes.

## Environment

- Everything runs in the `tim` conda env (Python 3.11, torch 2.13 cu130).
  Do NOT create new envs or install BITS deps (they pin py3.10/torch2.4 and
  conflict); BITS stays `blocked_resource`.
- Run tests with: `PYTHONPATH=src /home/tim/miniconda3/envs/tim/bin/python -m pytest tests/ -q`
- The system `python` (miniconda base, py3.13) has no deps — never use it.
- Bulky resources (raw data, weights, vendor source) live under
  `/mnt/data/ats_resources` (`ATS_DATA_ROOT`). They never enter git.

## GPU discipline (调试不要占卡)

- Debugging, smoke tests, unit tests and visualization are CPU-only. The smoke
  scripts export `CUDA_VISIBLE_DEVICES=""`; keep it that way, and do the same
  for any ad-hoc debug command that imports torch.
- GPU use is limited to explicitly launched long jobs: the vLLM server
  (`scripts/gen_eval/01_serve_llm.sh`) and real `gen_eval` training runs.
  Never start the vLLM server just to debug parsing/loop logic — use
  `llm.transport: mock` (config/smoke/quick.yaml) instead.

## Project discipline

- All keys, paths, budgets and task definitions live in `config/*.yaml`
  (`default.yaml` base + per-kind overrides). Never hard-code paths or
  endpoints in `src/`.
- Every runnable command is a script under `scripts/<kind>/`; scripts source
  `scripts/_env.sh` and run python via `ats_run`/`ats_run_cli`. Nothing runs
  outside scripts/ except the raw CLI for debugging.
- Experiment kinds: `smoke` | `gen_eval` | `perf` | `interp`. Logs go to
  `log/<kind>/`, artifacts to `results/<kind>/`, figures to `vis/generated/`.
  Only small `results/smoke` + `log/smoke` artifacts are committed.
- LLM endpoint: local vLLM `http://127.0.0.1:8848/v1` (Qwen3.5-4B), overridable
  via `ATS_LLM_BASE_URL` / `ATS_LLM_API_KEY`. No secret is ever committed.

## Honesty rules (from DESIGN.md — do not weaken)

- Unavailable things report structured, truthful status: heavy 2026 models are
  `not_vendored` (selection fails with reason + source URL), BITS is
  `blocked_resource`, MIMIC-IV is `blocked_access`, TSci is `unavailable`,
  fault injection raises `NotImplementedError`. Never fake, stub-score, or
  silently substitute.
- Failures never score as zero error; empty groups return `null` with `n=0`.
- Information boundaries are load-bearing: the controller only ever sees the
  `search` split; `confirm`/`test` stay behind the deterministic finalizer;
  no figure or feature may use post-cutoff information. `cli validate` and
  `tests/test_core.py` guard these — run both after any change to data,
  transforms, evaluator or harness.
- Models predict in ORIGINAL units; the evaluator normalizes with train-fitted
  zscore stats. Window `x` follows `DataView.normalize`, so value-copy models
  must inverse-transform via `model._norm`.

## Definition of done for a change

1. `PYTHONPATH=src .../envs/tim/bin/python -m pytest tests/ -q` passes.
2. `scripts/smoke/run_smoke.sh` completes and metrics are sane
   (sanity ladder: MeanValue/LastValue beatable, DLinear best on physiome).
3. Committed smoke artifacts under `results/smoke` are regenerated when
   behavior changes them.
4. README.md / DESIGN.md appendix updated if implementation status changed.
