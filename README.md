# Agent4TS

Minimal implementation of the DESIGN.md spec: a single-agent system that reads
data semantics and time-series plots, picks data-processing and forecasting
modules from a restricted action space, iterates on execution feedback, and
exports a frozen pipeline that predicts offline (no LLM at inference).

## Layout

```
config/      all keys, paths, budgets, task definitions (never hard-coded)
  default.yaml           base: paths, LLM endpoint, budgets, eval settings
  smoke/quick.yaml       tiny offline smoke config (mock LLM)
  gen_eval/main.yaml     full task set, real budgets, live LLM
scripts/     every runnable command lives here, nothing runs outside scripts/
  smoke/     run_smoke.sh, run_llm_smoke.sh, run_vis.sh
  gen_eval/  00_download_resources.sh … 11_tsci.sh (numbered pipeline)
  perf/      01_benchmark_perf.sh        (performance/cost experiments)
  interp/    01_trace_analysis.sh        (interpretability experiments)
log/         all logs, per experiment kind (smoke/ gen_eval/ perf/ interp/)
results/     all artifacts: traces, metrics json, csv tables, exported pipelines
vis/         visualization tooling; generated figures land in vis/generated/
src/agent4ts/  the package (schemas, data, models, transforms, observe,
               controller, harness, search, evaluator, baselines, faults, cli)
tests/       pytest suite over information boundaries, masks, splits, scoring
resources.lock.json  upstream versions, download sources, checksums
```

Bulky resources (raw data, model weights, vendor code) live outside the repo in
`/mnt/data/ats_resources` (override with `ATS_DATA_ROOT`). They never enter git;
only the small `results/smoke` + `log/smoke` artifacts are committed as living
proof the loop runs.

## Quick start

```bash
# 0) resources (P12 ~6.6MB; bits = source snapshot only; timeimm = EPA CSVs)
scripts/gen_eval/00_download_resources.sh p12

# 1) offline smoke test: mock LLM, full loop, ~1 min on CPU
scripts/smoke/run_smoke.sh

# 2) serve Qwen3.5-4B locally (background) and smoke the real controller
scripts/gen_eval/01_serve_llm.sh &          # vLLM on 127.0.0.1:8848
scripts/smoke/run_llm_smoke.sh

# 3) full pipeline
scripts/gen_eval/02_prepare.sh
scripts/gen_eval/03_validate.sh
scripts/gen_eval/04_benchmark.sh
scripts/gen_eval/05_search.sh --tasks p12_native --seeds 0,1,2
scripts/gen_eval/06_evaluate.sh --tasks p12_native --seed 0
scripts/gen_eval/07_predict.sh --pipeline results/gen_eval/search/p12_native/seed0/best_pipeline.json --tasks p12_native

# baselines / ablations / robustness
scripts/gen_eval/08_random_search.sh
scripts/gen_eval/09_degradation.sh
scripts/gen_eval/10_ablation.sh
scripts/gen_eval/11_tsci.sh

# other experiment kinds
scripts/perf/01_benchmark_perf.sh
scripts/interp/01_trace_analysis.sh

# visualization (figures + static HTML index)
scripts/smoke/run_vis.sh smoke        # or gen_eval
```

Everything runs in the `tim` conda env (Python 3.11, torch 2.13). The CLI is
`python -m agent4ts.cli {prepare,benchmark,search,evaluate,predict,validate}`
with `--config <stem>` / `--run-dir` / `--kind`.

## Honest implementation status

Working end to end (verified by `scripts/smoke/run_smoke.sh`):

* Event-based data contract (`Event = (entity, event_time, available_time,
  variable, value)`), real-time masks, entity-level 60/20/10/10 splits,
  no-leakage / padding-invariance / split-isolation checks (`cli validate`).
* Backends: **P12 Set A** (native irregular, our 24h->6h bucketed protocol, not
  the official challenge task), **Physiome-ODE** (3 named models, local RK4
  re-implementation, mother-trajectory isolation), **Time-IMM EPA-Air** loader
  (needs the CSVs downloaded first), **controlled degradations**
  (drop random/span, variable subsampling, noise — one at a time).
* Models runnable in `tim`: LastValue, MeanValue (sanity), DLinear, PatchTST-style,
  iTransformer-style (faithful re-implementations, not author code).
* Agent loop: statistics + semantic card + real PNGs attached to the multimodal
  LLM; one JSON action per round; validation, caching, budgets, accept/fallback,
  `trace.jsonl`, export of `best_pipeline.json`; deterministic confirm->test
  finalizer isolated from the agent.
* Baselines: random search (same action space/budget), no_vision / no_semantics /
  one_shot ablations, degradation sweep, perf + interpretability runners.

Explicitly **not** implemented (reported as such at runtime, never faked):

* **BITS runtime**: upstream pins Python 3.8-3.10 + torch 2.4-2.5, which
  conflicts with the `tim` env. The vendored source snapshot is inspected and
  `load_bits_task` returns `blocked_resource`. Track A runs on the native P12
  loader instead (per 2026-10-06 decision).
* **Heavy 2026 models** (TFMixer, KAFNet, APN, ASTGI, TiWeaver, HyperIMTS,
  tPatchGNN): author code not vendored; the registry lists them with
  `status: not_vendored` and `SELECT_MODEL` on them fails the model_runnable
  check with the reason + source URL. APN was the intended `f0`; DLinear is the
  recorded engineering stand-in.
* **TSci-adapted**: upstream TimeSeriesScientist is not vendored; the runner
  writes `status: unavailable` with setup steps instead of imitating TSci.
* **MIMIC-IV**: `blocked_access` until credentialed access exists.
* **RepoHealth**: not audited, hence not registered.
* **Fault injection/detection**: disabled placeholder (`faults.py`), per design.

## Keys & endpoints

`config/default.yaml -> llm` points at a local vLLM instance
(`http://127.0.0.1:8848/v1`, model `Qwen3.5-4B` from `/mnt/data/Qwen3.5-4B`).
Override with env vars `ATS_LLM_BASE_URL` / `ATS_LLM_API_KEY`, or per-experiment
config in `config/<kind>/*.yaml`. No secret is ever committed.
