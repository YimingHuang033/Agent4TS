# Agent4TS

A single-agent search system for irregular time-series forecasting pipelines.
The controller reads statistics, semantic cards and actual PNGs, proposes bounded
pipeline changes, and learns from search-split feedback. Frozen prediction uses
saved numerical models and never invokes an LLM or refits weights.

## Execution environment

Work through SSH on `BBQCard2`, project `/home/tim/Proj/ats`, existing `tim` conda
environment. Tests and smoke are CPU-only. Formal numerical-model training may
use GPU 0. **LLM inference is remote only**, using the Ark coding API and
`deepseek-v4.1-flash-economy`; text and image input have been verified. The old
local-vLLM launcher is disabled. Credentials are sourced from
`~/.config/agent4ts/ark.env` (mode 600), outside this repository.

```bash
# CPU engineering smoke
bash scripts/smoke/run_smoke.sh
# CPU real-API controller smoke
bash scripts/smoke/run_llm_smoke.sh
# Resumable formal matrix
bash scripts/gen_eval/13_complete.sh
# Read-only current progress
/home/tim/miniconda3/envs/tim/bin/python scripts/gen_eval/status_complete.py
```

## Current implementation

- Metric v2: entity-variable macro normalized MAE/MSE, raw per-variable errors,
  coverage and train-fitted long-gap/stale/density/asynchrony groups.
- Search, confirm and test caches are separate; private splits belong to the
  deterministic finalizer. Candidates retain fitted models across confirmation
  and testing. Export includes a fitted model, normalizer and checksum.
- Frozen prediction rejects old config-only artifacts. Pickle artifacts are
  trusted local files and never committed.
- Random and agent search share a finite action space. Invalid/duplicate actions,
  actual API retries/token usage, timeouts and failures are recorded.
- Seven upstream adapters are runnable: APN, ASTGI, TFMixer, KAFNet, HyperIMTS,
  TiWeaver and tPatchGNN. Their source revisions and compatibility patches are in
  `resources.lock.json`; these are project training adapters, not official-paper
  training reproductions. Existing DLinear, PatchTST-style, iTransformer-style,
  LastValue and MeanValue bring the registry to 12 runnable entries.
- Upstream irregular adapters optionally consume explicit historical age
  covariates, scaled by known history duration; unsupported models reject that
  action. The feature never uses post-cutoff observations.
- TSci-adapted reuses upstream analysis/selection prompts and actual validation
  run/test/select methods. Transport and execution are bridged to the shared
  harness; synthetic fallback scores are prohibited. Report generation is omitted
  and preprocessing uses the audited task views.

## Data and validation

P12 native/custom forecasting tasks, all three named Physiome mechanisms, and
EPA-Air Los Angeles/Dallas/Denver pass data checks. EPA uses verified source
column mapping, calendar alignment and disjoint chronological windows. Unknown
physical units remain explicitly unknown, with no guessed conversion.

**24 core tests pass.** All 12 models pass CPU training, fitted export/reload and
future-label-independence checks. All seven upstream adapters also pass those
checks with age features enabled. Reports are in
`results/smoke/upstream_adapter_checks/report.json` and
`results/smoke/upstream_age_checks/report.json`. The real API controller and
TSci-adapted have completed smoke runs.

## Formal experiments — running, not complete

Current artifacts: `results/gen_eval/v2_ark_20261107/`.
Log: `log/gen_eval/v2_ark_20261107.log`.

The 720-cell matrix covers eight accessible tasks and three seeds: 12 fixed
models, random search, agent, no_vision/no_semantics/one_shot and TSci-adapted;
plus APN/DLinear/random/agent across eight degradations of each Physiome task.
APN is the frozen initial predictor for this matrix. Old smoke fixtures retain
DLinear for regression continuity. Each cell has an explicit result/status;
code hashes, configuration and LLM identity are recorded. A DONE marker only
means execution ended: inspect failures and summary support counts before
interpreting results. Preflight directories and old v1 scores are separate.

## Remaining limitations

- MIMIC-IV: explicitly deferred by the user; no credentialed data available.
- RepoHealth: source fetched and audited; React has 1 negative open_prs row and
  137 negative open_issues rows. Initial-stock semantics and future interval
  availability remain unresolved. Status: `blocked_semantics`; no clipping or
  fabricated stock reconstruction.
- TSRouter: source downloaded, but its TSRBench correctness labels and
  task/query/modality/model embeddings have no validated numerical-forecasting
  bridge. Status: `blocked_protocol`; not represented as a completed baseline.
- BITS runtime and its full official suite remain unavailable under the retained
  environment constraint. Reusing its model sources does not reproduce its
  official evaluation protocol.
- Fault injection and unrestricted function-structure search remain outside the
  agreed first-version scope.

The full research request is not complete while the matrix and protocol/semantic
blockers remain outstanding. See `DESIGN.md` for the specification and dated
implementation record, and `AGENTS.md` for operational constraints.
MD'
### 2026-11-09 expansion preparation (CPU validation continued on Nov 10)

`config/gen_eval/expand_v3.yaml` defines two **custom forecasting subsets**, not
BITS benchmark scores. The pinned BITS USHCN/PAMAP2 tensors are downloaded and
retained for audit, but not used for formal experiments: USHCN upstream applies
whole-data normalization, while the PAMAP2 tensor's channel/subject lineage is
not recoverable. The replacements are 20 NOAA HCN-flag stations (1990–1999,
720 disjoint 100-day windows) and all nine official PAMAP2 Protocol subjects
(168 prespecified 10-second windows, five raw sensor channels). Splits keep each
mother station/subject together. No interpolation or loader normalization occurs.

Run `bash scripts/gen_eval/14_prepare_expand.sh` to verify/recover resources and
write audits in `results/smoke/expand_1109/`. `check_expand.py` and
`check_expand_api.py` under `scripts/smoke` validate CPU model adapters, frozen
reload, label independence and live Ark calls. See `agent_notes/1109_expand.md`
for the validated administrator handoff and remaining formal experiments.

V3 uses canonical model defaults and one finite action-space implementation for
agent/random/greedy/TSci model comparisons. B includes f0; failed training counts;
patience is disabled. `15_run_expand.sh manifest` freezes identities and source
snapshots, then `15_run_expand.sh <phase>` runs an explicit phase. Failures require
an explicit JSON retry list and retain previous attempts. Status/summary scripts
read the manifest, including pending cells and first-attempt success rates.

V3 cannot be pooled with v2: grid aggregation now uses an actual arithmetic mean
and retains cutoff observations; history gap statistics are computed within each
entity and carry explicit time units. Ark `enable_thinking: false` is now sent as
`thinking.type=disabled`; all policies share the same one-retry transport. LLMs
use remote API only. GPU is reserved for administrator-launched numerical jobs.
