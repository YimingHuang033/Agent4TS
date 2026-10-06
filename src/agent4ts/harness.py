"""Harness: validate -> execute -> cache -> score -> archive -> fallback.

This is the deterministic core the agent is wrapped around. It owns the budget,
the search/confirm/test isolation, the archive and the trace. The controller
never sees confirm or test results.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import evaluator
from .logging_utils import get_logger
from .models.registry import CATALOG, make as make_model, runnable_ids
from .schemas import Action, DataView, Feedback, PipelineSpec, TaskBundle
from .transforms import Normalizer, build_labeled_windows, compatible

log = get_logger("harness")

ALLOWED_OPS = ["SELECT_DATA", "SET_TRANSFORM", "SELECT_MODEL", "SET_MODEL_OPTION", "INSPECT", "STOP"]


class BudgetExhausted(RuntimeError):
    pass


class Harness:
    def __init__(self, bundle: TaskBundle, cfg: dict[str, Any], run_dir: Path,
                 seed: int = 0, allow_test: bool = False):
        self.bundle = bundle
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.seed = seed
        self.allow_test = allow_test
        self.budget = dict(cfg["budget"])
        self.used = {"trials": 0, "llm_calls": 0, "gpu_hours": 0.0}
        self.archive: list[dict[str, Any]] = []
        self.cache: dict[str, Feedback] = {}
        self.trace_path = self.run_dir / "trace.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_feedback: Feedback | None = None
        self.incumbent: str | None = None

        # split windows + a normalizer fitted on TRAIN only
        self.eval_cfg = cfg["evaluation"]
        self.norm = Normalizer.fit(bundle.history, bundle.entities_for("train"),
                                   cfg.get("normalize_mode", "zscore"))
        self._windows: dict[str, Any] = {}
        self.f0_id: str | None = None

    # ------------------------------------------------------------- data views
    def windows(self, split: str, view: DataView):
        key = (split, json.dumps(view.to_dict(), sort_keys=True))
        if key not in self._windows:
            self._windows[key] = build_labeled_windows(self.bundle, view, split, self.norm)
        return self._windows[key]

    def default_view(self) -> DataView:
        return DataView()

    # ------------------------------------------------------------- validation
    def validate(self, action: Action, parent: PipelineSpec) -> list[dict[str, Any]]:
        """Return structured check results; a hard failure means do not train."""
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, msg: str = "") -> None:
            checks.append({"check": name, "ok": bool(ok), "msg": msg})

        add("op_whitelisted", action.op in ALLOWED_OPS,
            f"op={action.op} allowed={ALLOWED_OPS}")

        if action.op == "SELECT_MODEL":
            ok = action.value in CATALOG
            add("model_registered", ok, f"value={action.value}")
            if ok and CATALOG[action.value]["factory"] is None:
                add("model_runnable", False,
                    f"{action.value} not vendored: {CATALOG[action.value].get('blocked_reason')}")
            elif ok:
                add("model_runnable", True, "")
        elif action.op == "SET_MODEL_OPTION":
            meta = CATALOG[parent.model_id]
            ok_key = action.target_field in meta.get("options", {})
            add("option_registered", ok_key, f"field={action.target_field}")
            if ok_key:
                allowed = meta["options"][action.target_field]
                add("option_value_allowed", (not allowed) or action.value in allowed,
                    f"value={action.value} allowed={allowed}")
        elif action.op == "SET_TRANSFORM":
            ok_field = action.target_field in DataView().__dict__
            add("transform_field", ok_field, f"field={action.target_field}")
            if ok_field:
                trial = DataView(**parent.data_view.to_dict())
                setattr(trial, action.target_field, action.value)
                ok, msg = compatible(self.bundle, trial)
                add("transform_compatible", ok, msg)
        elif action.op == "SELECT_DATA":
            ok_field = action.target_field in DataView().__dict__
            add("data_field", ok_field, f"field={action.target_field}")

        return checks

    def checks_ok(self, checks: list[dict[str, Any]]) -> bool:
        return all(c["ok"] for c in checks)

    # ------------------------------------------------------------- run a trial
    def run_trial(self, spec: PipelineSpec, split: str = "search") -> Feedback:
        fp = spec.fingerprint(self._upstream_hash(), self.bundle.spec.dataset_version,
                              self._splits_hash())
        if fp in self.cache:
            log.info("cache hit %s (%s)", spec.pipeline_id, fp)
            fb = self.cache[fp]
            fb.checks = fb.checks + [{"check": "cache", "ok": True, "msg": fp}]
            spec.pipeline_id = spec.pipeline_id  # keep id
            return fb

        if self.used["trials"] >= self.budget["trials"]:
            return Feedback(status="rejected", checks=[{"check": "budget", "ok": False,
                            "msg": "trial budget exhausted"}],
                            remaining_budget=self.remaining())

        t0 = time.time()
        view = spec.data_view
        ok, msg = compatible(self.bundle, view)
        if not ok:
            return Feedback(status="rejected",
                            checks=[{"check": "compat", "ok": False, "msg": msg}],
                            remaining_budget=self.remaining())

        self.used["trials"] += 1
        try:
            train = self.windows("train", view)
            evalset = self.windows(split, view)
            if not train or not evalset:
                raise RuntimeError(f"empty windows (train={len(train)} eval={len(evalset)})")
            model = make_model(spec.model_id, {**spec.model_config, "seed": spec.seed},
                               self.cfg.get("resources"))
            grid = np.unique(np.concatenate([lw.query_times for lw in evalset]))
            var_ids = np.arange(len(self.bundle.variable_names))
            # give the model the view so its lookback grid matches this data_view
            model._view = view
            model.in_grid = None
            model.fit(train, self.bundle, grid, var_ids)
            preds = [model.predict(lw) for lw in evalset]
            res = evaluator.score_windows(evalset, preds, self.bundle, self.norm, self.eval_cfg)
            fb = Feedback(status="ok", checks=[{"check": "executed", "ok": True, "msg": ""}],
                          metrics=res["metrics"], groups=res["groups"],
                          support_counts=res["support_counts"],
                          cost={"wall_s": round(time.time() - t0, 3)},
                          remaining_budget=self.remaining())
            fb._per_variable = res["per_variable"]  # type: ignore[attr-defined]
            fb._artifacts = {"grid": grid.tolist(), "preds": [p.tolist() for p in preds]}
        except NotImplementedError as exc:
            fb = Feedback(status="error", error=f"not_implemented: {exc}",
                          checks=[{"check": "model_available", "ok": False, "msg": str(exc)}],
                          remaining_budget=self.remaining())
        except Exception as exc:  # training/format failures never fake a score
            log.warning("trial failed: %s", exc)
            fb = Feedback(status="error", error=f"{type(exc).__name__}: {exc}",
                          checks=[{"check": "executed", "ok": False, "msg": str(exc)}],
                          cost={"wall_s": round(time.time() - t0, 3)},
                          remaining_budget=self.remaining())

        self.cache[fp] = fb
        spec_fp = fp
        entry = {"pipeline": spec.to_dict(), "fingerprint": spec_fp,
                 "status": fb.status, "metrics": fb.metrics,
                 "error": fb.error, "wall_s": fb.cost.get("wall_s")}
        self.archive.append(entry)
        self._write_trace({"event": "trial", "split": split, **entry})
        self.last_feedback = fb
        return fb

    # ------------------------------------------------------------- accept/rlb
    def update_incumbent(self, spec: PipelineSpec, fb: Feedback) -> tuple[bool, float | None]:
        """Accept if checks pass and primary metric improved. Ties -> cheaper wins."""
        if fb.status != "ok":
            return False, None
        new = fb.metrics.get(self.eval_cfg["primary_metric"])
        if new is None:
            return False, None
        if self.incumbent is None:
            self.incumbent = spec.pipeline_id
            return True, None
        cur = self._incumbent_metric()
        if cur is None or new < cur:
            delta = evaluator.improvement(cur, new)
            self.incumbent = spec.pipeline_id
            fb.delta_vs_parent = delta
            return True, delta
        return False, evaluator.improvement(cur, new)

    def _incumbent_metric(self) -> float | None:
        for e in reversed(self.archive):
            if e["pipeline"]["pipeline_id"] == self.incumbent and e["status"] == "ok":
                return e["metrics"].get(self.eval_cfg["primary_metric"])
        return None

    def f0_metric(self) -> float | None:
        for e in self.archive:
            if e["pipeline"]["pipeline_id"] == self.f0_id and e["status"] == "ok":
                return e["metrics"].get(self.eval_cfg["primary_metric"])
        return None

    # ------------------------------------------------------------- budgets
    def remaining(self) -> dict[str, Any]:
        return {
            "trials": self.budget["trials"] - self.used["trials"],
            "llm_calls": self.budget["llm_calls"] - self.used["llm_calls"],
            "gpu_hours": round(self.budget["gpu_hours"] - self.used["gpu_hours"], 4),
        }

    def exhausted(self) -> bool:
        r = self.remaining()
        return r["trials"] <= 0 or r["llm_calls"] <= 0 or r["gpu_hours"] <= 0

    def spend_llm_call(self) -> None:
        self.used["llm_calls"] += 1

    def add_gpu_time(self, hours: float) -> None:
        self.used["gpu_hours"] += hours

    # ------------------------------------------------------------- testing
    def evaluate_test(self, spec: PipelineSpec) -> Feedback:
        """Only ever called by the deterministic finalizer, never during search."""
        return self.run_trial(spec, split="test")

    # ------------------------------------------------------------- export
    def export(self, spec: PipelineSpec, out: Path) -> dict[str, Any]:
        out.mkdir(parents=True, exist_ok=True)
        best = {
            "best_pipeline": spec.to_dict(),
            "normalizer": self.norm.to_dict(),
            "metric_version": self.eval_cfg["metric_version"],
            "upstream_hash": self._upstream_hash(),
        }
        (out / "best_pipeline.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        return best

    # ------------------------------------------------------------- helpers
    def _upstream_hash(self) -> str:
        import hashlib
        lock = Path(self.cfg["paths"]["resource_lock"])
        payload = lock.read_bytes() if lock.exists() else b"no-lock"
        return hashlib.sha256(payload).hexdigest()[:16]

    def _splits_hash(self) -> str:
        import hashlib
        payload = json.dumps({k: v for k, v in self.bundle.spec.split_ids.items()}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _write_trace(self, record: dict[str, Any]) -> None:
        record = {**record, "ts": time.time(), "seed": self.seed,
                  "task_id": self.bundle.spec.task_id}
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def write_trace(self, record: dict[str, Any]) -> None:
        self._write_trace(record)
