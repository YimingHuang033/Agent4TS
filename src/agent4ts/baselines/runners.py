"""Standalone experiment runners referenced by scripts/.

Each takes --config/--kind/--run-dir and writes into results/<kind>/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import config as cfgmod
from ..data import load_task
from ..logging_utils import get_logger, setup_logging

log = get_logger("runners")

F0_ANCHOR = "physiome_dupont"


def _load(cfg, task_key):
    return load_task(cfg["tasks"][task_key], cfg)


def _pick_key(cfg, requested: str | None) -> str:
    if requested:
        return requested
    if F0_ANCHOR in cfg.get("tasks", {}):
        return F0_ANCHOR
    return sorted(cfg["tasks"])[0]


# ------------------------------------------------------------------ random

def run_random(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--kind", default="gen_eval")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = cfgmod.load_config(a.config)
    setup_logging(cfgmod.log_dir(cfg, a.kind) / "random_search.log")
    from .random_search import run_random_search
    key = _pick_key(cfg, a.tasks)
    out_dir = cfgmod.experiment_dir(cfg, a.kind) / "random_search" / key
    bundle = _load(cfg, key)
    res = run_random_search(bundle, cfg, out_dir, seed=a.seed)
    log.info("random search %s: incumbent nmae=%s f0=%s",
             key, res["incumbent_metric"], res["f0_metric"])
    return 0


# ------------------------------------------------------------------ degradation

def run_degradation(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--kind", default="gen_eval")
    ap.add_argument("--task", default=None)
    ap.add_argument("--model", default="DLinear")
    a = ap.parse_args(argv)
    cfg = cfgmod.load_config(a.config)
    setup_logging(cfgmod.log_dir(cfg, a.kind) / "degradation.log")
    from ..degradation import apply_degradation, default_grid
    from ..harness import Harness
    from ..schemas import PipelineSpec

    key = _pick_key(cfg, a.task)
    bundle = _load(cfg, key)
    out_dir = cfgmod.experiment_dir(cfg, a.kind) / "degradation" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, spec in enumerate(default_grid()):
        deg = apply_degradation(bundle, spec, seed=0)
        h = Harness(deg, cfg, out_dir / spec["tag"], seed=0)
        from ..schemas import DataView
        ps = PipelineSpec(pipeline_id=f"deg_{spec['tag']}", data_view=DataView(),
                          model_id=a.model, seed=0)
        fb = h.run_trial(ps, split="search")
        rows.append({"tag": spec["tag"], "kind": spec["kind"], "setting": spec,
                     "status": fb.status, "normalized_mae": fb.metrics.get("normalized_mae"),
                     "n_valid": fb.metrics.get("n_valid_targets"), "error": fb.error})
        log.info("degradation %s: %s nmae=%s", spec["tag"], fb.status,
                 fb.metrics.get("normalized_mae"))
    # undegraded reference
    h0 = Harness(bundle, cfg, out_dir / "baseline", seed=0)
    from ..schemas import DataView
    fb0 = h0.run_trial(PipelineSpec(pipeline_id="deg_baseline", data_view=DataView(),
                                    model_id=a.model, seed=0), split="search")
    rows.insert(0, {"tag": "none", "kind": "baseline", "setting": {},
                    "status": fb0.status, "normalized_mae": fb0.metrics.get("normalized_mae"),
                    "n_valid": fb0.metrics.get("n_valid_targets"), "error": fb0.error})
    (out_dir / "degradation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _csv(out_dir / "degradation.csv", rows)
    return 0


# ------------------------------------------------------------------ tsci

def run_tsci(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--kind", default="gen_eval")
    a = ap.parse_args(argv)
    cfg = cfgmod.load_config(a.config)
    setup_logging(cfgmod.log_dir(cfg, a.kind) / "tsci.log")
    from . import tsci
    st = tsci.status(cfg.get("resources"))
    out_dir = cfgmod.experiment_dir(cfg, a.kind) / "tsci"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    log.info("TSci status: %s (%s)", st["status"], st.get("blocked_reason", ""))
    if st["status"] != "active":
        log.warning("TSci not runnable; wrote status.json. See baselines/tsci.py.")
        return 0
    from .tsci import run
    run(bundle=_load(cfg, _pick_key(cfg, None)), cfg=cfg, run_dir=out_dir)
    return 0


# ------------------------------------------------------------------ perf

def run_perf(argv: list[str] | None = None) -> int:
    """Cost eval: per-model wall time and (if available) peak GPU memory."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--kind", default="perf")
    ap.add_argument("--task", default=None)
    ap.add_argument("--models", default="LastValue,MeanValue,DLinear,PatchTST,iTransformer")
    a = ap.parse_args(argv)
    cfg = cfgmod.load_config(a.config)
    setup_logging(cfgmod.log_dir(cfg, a.kind) / "perf_benchmark.log")
    from ..harness import Harness
    from ..schemas import DataView, PipelineSpec

    key = _pick_key(cfg, a.task)
    bundle = _load(cfg, key)
    out_dir = cfgmod.experiment_dir(cfg, a.kind) / "benchmark" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    rows = []
    for mid in a.models.split(","):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        h = Harness(bundle, cfg, out_dir / mid, seed=0)
        ps = PipelineSpec(pipeline_id=f"perf_{mid}", data_view=DataView(), model_id=mid, seed=0)
        t0 = time.time()
        fb = h.run_trial(ps, split="search")
        wall = time.time() - t0
        peak = 0.0
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e6
        rows.append({"model": mid, "status": fb.status, "wall_s": round(wall, 3),
                     "peak_gpu_mb": round(peak, 1),
                     "nmae": fb.metrics.get("normalized_mae") if fb.metrics else None,
                     "error": fb.error})
        log.info("perf %s: %.2fs peak=%.1fMB", mid, wall, peak)
    (out_dir / "perf.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _csv(out_dir / "perf.csv", rows)
    return 0


# ------------------------------------------------------------------ interp

def run_interp(argv: list[str] | None = None) -> int:
    """Interpretability: summarise action traces from the latest search run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--kind", default="interp")
    ap.add_argument("--search-kind", default="gen_eval")
    ap.add_argument("--task", default=None)
    a = ap.parse_args(argv)
    cfg = cfgmod.load_config(a.config)
    setup_logging(cfgmod.log_dir(cfg, a.kind) / "trace_analysis.log")
    key = _pick_key(cfg, a.task)
    search_root = cfgmod.experiment_dir(cfg, a.search_kind) / "search" / key
    out_dir = cfgmod.experiment_dir(cfg, a.kind) / key
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = sorted(search_root.rglob("trace.jsonl")) if search_root.exists() else []
    if not traces:
        log.warning("no traces under %s; run a search first", search_root)
        (out_dir / "trace_analysis.json").write_text(
            json.dumps({"status": "no_traces", "searched": str(search_root)}, indent=2))
        return 0

    summary = []
    for tp in traces:
        events = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
        acts = [e for e in events if e.get("event") in ("propose", "rejected", "feedback", "stop")]
        summary.append({
            "trace": str(tp),
            "n_events": len(events),
            "n_proposals": sum(1 for e in events if e.get("event") == "propose"),
            "n_rejected": sum(1 for e in events if e.get("event") == "rejected"),
            "n_accepted": sum(1 for e in events if e.get("event") == "feedback" and e.get("accepted")),
            "ops": _op_histogram(events),
            "figures_seen": sum(len(e.get("figures", [])) for e in events if e.get("event") == "propose"),
        })
    (out_dir / "trace_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _csv(out_dir / "trace_analysis.csv", summary)
    log.info("analysed %d traces -> %s", len(summary), out_dir)
    return 0


def _op_histogram(events) -> dict:
    hist: dict[str, int] = {}
    for e in events:
        if e.get("event") == "propose":
            op = (e.get("action") or {}).get("op", "?")
            hist[op] = hist.get(op, 0) + 1
    return hist


def _csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
