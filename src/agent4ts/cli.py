"""Command line: prepare | benchmark | search | evaluate | predict | validate.

Every entry point takes --config <stem> and --run-dir. Nothing here hard-codes a
path; the config module resolves config/ paths. All commands are meant to be run
through scripts/ (see scripts/*/*.sh), but the CLI is directly usable too.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import config as cfgmod
from . import search as search_mod
from .harness import Harness
from .logging_utils import get_logger, setup_logging
from .models.registry import availability_report, runnable_ids
from .schemas import PipelineSpec, TaskBundle
from .transforms import Normalizer, build_labeled_windows

log = get_logger("cli")


def _load_bundle(cfg: dict, task_key: str) -> TaskBundle:
    spec = cfg["tasks"][task_key]
    from .data import load_task
    return load_task(spec, cfg)


def _task_keys(cfg: dict) -> list[str]:
    return sorted(cfg.get("tasks", {}).keys())


# --------------------------------------------------------------------- prepare

def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = cfgmod.load_config(args.config, args.extra)
    run_dir = Path(args.run_dir) if args.run_dir else cfgmod.experiment_dir(cfg, args.kind) / "prepare"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfgmod.log_dir(cfg, args.kind) / "prepare.log")
    log.info("prepare: config=%s run_dir=%s", args.config, run_dir)

    out: dict = {"runnable_models": runnable_ids(), "availability": availability_report(cfg.get("resources"))}
    out["tasks"] = {}
    for key in _task_keys(cfg):
        entry: dict = {"spec": cfg["tasks"][key]}
        try:
            bundle = _load_bundle(cfg, key)
            entry["status"] = "ok"
            entry["n_entities"] = {k: len(v) for k, v in bundle.spec.split_ids.items()}
            entry["n_history_events"] = int(len(bundle.history))
            entry["n_targets"] = int(len(bundle.targets))
            entry["variables"] = bundle.variable_names
            entry["card_backend"] = bundle.card.backend
        except Exception as exc:
            entry["status"] = "blocked"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["as_status"] = getattr(exc, "as_status", lambda: None)()
            log.warning("task %s blocked: %s", key, exc)
        out["tasks"][key] = entry

    (run_dir / "prepare.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("prepare done: %d/%d tasks ok", sum(1 for t in out["tasks"].values() if t["status"] == "ok"),
             len(out["tasks"]))
    print(json.dumps({"run_dir": str(run_dir), "tasks": {k: v["status"] for k, v in out["tasks"].items()}},
                     indent=2))
    return 0


# ------------------------------------------------------------------- benchmark

def cmd_benchmark(args: argparse.Namespace) -> int:
    """Train fixed models (no agent) and report per-model cost/metrics."""
    cfg = cfgmod.load_config(args.config, args.extra)
    run_dir = Path(args.run_dir) if args.run_dir else cfgmod.experiment_dir(cfg, args.kind) / "benchmark"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfgmod.log_dir(cfg, args.kind) / "benchmark.log")
    models = args.models.split(",") if args.models else runnable_ids()
    keys = args.tasks.split(",") if args.tasks else _task_keys(cfg)

    results = {}
    for key in keys:
        try:
            bundle = _load_bundle(cfg, key)
        except Exception as exc:
            log.warning("task %s blocked: %s", key, exc)
            results[key] = {"status": "blocked", "error": str(exc)}
            continue
        results[key] = {}
        for model_id in models:
            h = Harness(bundle, cfg, run_dir / key / model_id, seed=args.seed)
            spec = PipelineSpec(pipeline_id=f"bench_{model_id}", data_view=search_mod.initial_spec(bundle).data_view,
                                model_id=model_id, seed=args.seed)
            fb = h.run_trial(spec, split="search")
            h.export(spec, run_dir / key / model_id)
            results[key][model_id] = {"status": fb.status, "metrics": fb.metrics,
                                      "groups": fb.groups, "cost": fb.cost, "error": fb.error}
            log.info("benchmark %s/%s: %s nmae=%s", key, model_id, fb.status,
                     fb.metrics.get("normalized_mae"))
    (run_dir / "benchmark.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_csv(run_dir / "benchmark.csv", results)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


# ---------------------------------------------------------------------- search

def cmd_search(args: argparse.Namespace) -> int:
    cfg = cfgmod.load_config(args.config, args.extra)
    run_dir = Path(args.run_dir) if args.run_dir else cfgmod.experiment_dir(cfg, args.kind) / "search"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfgmod.log_dir(cfg, args.kind) / "search.log")
    if args.ablation:
        run_dir = run_dir / args.ablation
    keys = args.tasks.split(",") if args.tasks else _task_keys(cfg)

    all_out = {}
    for key in keys:
        for seed in [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]:
            try:
                bundle = _load_bundle(cfg, key)
            except Exception as exc:
                log.warning("task %s blocked: %s", key, exc)
                all_out[f"{key}_s{seed}"] = {"status": "blocked", "error": str(exc)}
                continue
            target = run_dir / key / f"seed{seed}"
            out = search_mod.run_search(bundle, cfg, target, seed=seed, ablation=args.ablation)
            all_out[f"{key}_s{seed}"] = out
            log.info("search %s seed=%d: rounds=%d incumbent=%s nmae=%s",
                     key, seed, out["rounds"], out["incumbent"]["pipeline_id"], out["incumbent_metric"])
    (run_dir / "search_all.json").write_text(json.dumps(all_out, indent=2, default=str), encoding="utf-8")
    _write_csv(run_dir / "search.csv", all_out)
    print(json.dumps({"run_dir": str(run_dir), "tasks": list(all_out)}, indent=2))
    return 0


# -------------------------------------------------------------------- evaluate

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Deterministic confirm+test on locked candidates from a search run."""
    cfg = cfgmod.load_config(args.config, args.extra)
    run_dir = Path(args.run_dir) if args.run_dir else cfgmod.experiment_dir(cfg, args.kind) / "search"
    setup_logging(cfgmod.log_dir(cfg, args.kind) / "evaluate.log")
    keys = args.tasks.split(",") if args.tasks else _task_keys(cfg)
    out = {}
    for key in keys:
        search_file = run_dir / key / f"seed{args.seed}" / "search_all.json"
        if not search_file.exists():
            # single-task search writes search.json history inside the run dir
            search_file = run_dir / "search_all.json"
        entry = _load_search_incumbent(search_file, key, args.seed)
        if entry is None:
            out[key] = {"status": "no_search_result"}
            continue
        bundle = _load_bundle(cfg, key)
        cands = [pipeline_from_dict(entry["incumbent"])]
        res = search_mod.finalize(bundle, cfg, run_dir / key / f"seed{args.seed}", cands, seed=args.seed)
        out[key] = res
        log.info("evaluate %s: %s", key, res.get("status"))
    (run_dir / "evaluate.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    _write_csv(run_dir / "evaluate.csv", out)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


def _load_search_incumbent(path: Path, key: str, seed: int):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if key in data:
        return data[key]
    for k, v in data.items():
        if isinstance(v, dict) and "incumbent" in v and key in k:
            return v
    for k, v in data.items():
        if isinstance(v, dict) and "incumbent" in v:
            return v
    return None


def pipeline_from_dict(d: dict) -> PipelineSpec:
    """Rehydrate a PipelineSpec whose data_view was serialised as a dict."""
    return PipelineSpec(
        pipeline_id=d["pipeline_id"],
        data_view=_dataview(d.get("data_view", {})),
        model_id=d["model_id"],
        model_config=d.get("model_config", {}),
        seed=d.get("seed", 0),
        parent_id=d.get("parent_id"),
    )


# --------------------------------------------------------------------- predict

def cmd_predict(args: argparse.Namespace) -> int:
    """Load a frozen best_pipeline.json and predict — no LLM involved."""
    cfg = cfgmod.load_config(args.config, args.extra)
    setup_logging(cfgmod.log_dir(cfg, args.kind) / "predict.log")
    pipeline_file = Path(args.pipeline)
    payload = json.loads(pipeline_file.read_text(encoding="utf-8"))
    spec = PipelineSpec(
        pipeline_id=payload["best_pipeline"]["pipeline_id"],
        data_view=_dataview(payload["best_pipeline"]["data_view"]),
        model_id=payload["best_pipeline"]["model_id"],
        model_config=payload["best_pipeline"].get("model_config", {}),
        seed=payload["best_pipeline"].get("seed", 0),
    )
    key = args.tasks if args.tasks else _task_keys(cfg)[0]
    bundle = _load_bundle(cfg, key)
    norm = Normalizer.from_dict(payload["normalizer"])
    from .models.registry import make as make_model
    train = build_labeled_windows(bundle, spec.data_view, "train", norm)
    test = build_labeled_windows(bundle, spec.data_view, "test", norm)
    import numpy as np
    grid = np.unique(np.concatenate([lw.query_times for lw in test]))
    model = make_model(spec.model_id, {**spec.model_config, "seed": spec.seed}, cfg.get("resources"))
    model.fit(train, bundle, grid, np.arange(len(bundle.variable_names)))
    preds = [model.predict(lw) for lw in test]
    out_dir = Path(args.run_dir) if args.run_dir else cfgmod.experiment_dir(cfg, args.kind) / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lw, p in zip(test, preds):
        for q in range(len(lw.query_times)):
            rows.append({"entity_id": lw.window.entity_id,
                         "query_time": float(lw.query_times[q]),
                         "variable": bundle.var_name(int(lw.query_var_ids[q])),
                         "prediction": float(p[q]),
                         "truth": float(lw.y_true[q]) if np.isfinite(lw.y_true[q]) else None})
    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_rows_csv(out_dir / "predictions.csv", rows)
    log.info("predict: %d query points -> %s", len(rows), out_dir)
    print(json.dumps({"run_dir": str(out_dir), "n_predictions": len(rows)}, indent=2))
    return 0


def _dataview(d):
    from .schemas import DataView
    return DataView(**{k: v for k, v in d.items() if k in DataView().__dict__})


# -------------------------------------------------------------------- validate

def cmd_validate(args: argparse.Namespace) -> int:
    """Sanity checks: no future leakage, padding-invariance, split isolation."""
    from .tests_checks import run_checks
    cfg = cfgmod.load_config(args.config, args.extra)
    setup_logging(cfgmod.log_dir(cfg, args.kind) / "validate.log")
    keys = args.tasks.split(",") if args.tasks else _task_keys(cfg)
    report = {}
    for key in keys:
        try:
            bundle = _load_bundle(cfg, key)
        except Exception as exc:
            report[key] = {"status": "blocked", "error": str(exc)}
            continue
        report[key] = run_checks(bundle, cfg)
    out_dir = Path(args.run_dir) if args.run_dir else cfgmod.experiment_dir(cfg, args.kind) / "validate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validate.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if all(r.get("status") != "fail" for r in report.values()) else 1


# ----------------------------------------------------------------------- utils

def _write_csv(path: Path, results: dict) -> None:
    rows = []
    for task, models in results.items():
        if not isinstance(models, dict):
            continue
        for model, r in models.items():
            if not isinstance(r, dict) or "metrics" not in r:
                continue
            m = r["metrics"] or {}
            rows.append({"task": task, "model": model, "status": r.get("status"),
                         "normalized_mae": m.get("normalized_mae"),
                         "normalized_mse": m.get("normalized_mse"),
                         "n_valid_targets": m.get("n_valid_targets")})
    _write_rows_csv(path, rows)


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    cols = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent4ts", description="Agent4TS minimal implementation")
    p.add_argument("--config", default="default", help="config stem under config/ (e.g. smoke/quick)")
    p.add_argument("--extra", nargs="*", default=None, help="extra config stems merged last")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--kind", default="gen_eval", help="experiment kind: smoke|gen_eval|perf|interp")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare"); sp.set_defaults(func=cmd_prepare)

    sb = sub.add_parser("benchmark")
    sb.add_argument("--models", default=None); sb.add_argument("--tasks", default=None)
    sb.add_argument("--seed", type=int, default=0); sb.set_defaults(func=cmd_benchmark)

    ss = sub.add_parser("search")
    ss.add_argument("--tasks", default=None); ss.add_argument("--seeds", default=None)
    ss.add_argument("--ablation", default=None, choices=[None, "no_vision", "no_semantics", "one_shot"])
    ss.set_defaults(func=cmd_search)

    se = sub.add_parser("evaluate")
    se.add_argument("--tasks", default=None); se.add_argument("--seed", type=int, default=0)
    se.set_defaults(func=cmd_evaluate)

    spr = sub.add_parser("predict")
    spr.add_argument("--pipeline", required=True); spr.add_argument("--tasks", default=None)
    spr.set_defaults(func=cmd_predict)

    sv = sub.add_parser("validate")
    sv.add_argument("--tasks", default=None); sv.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as exc:
        log.error("fatal: %s\n%s", exc, traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
