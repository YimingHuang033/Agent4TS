"""Search loop + deterministic finalizer.

`run_search` iterates: observe -> propose -> validate -> trial -> accept/fallback.
The controller only ever sees the `search` split. After search, `finalize`
locks at most 3 candidates and a deterministic program picks one on `confirm`,
then and only then scores `test` once, with no feedback returned to the agent.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import observe as observe_mod
from .controller import Controller
from .harness import Harness
from .logging_utils import get_logger
from .schemas import Action, DataView, Observation, PipelineSpec, TaskBundle

log = get_logger("search")

ABLATIONS = {"no_vision", "no_semantics", "one_shot"}


def make_pipeline_id(model_id: str, n: int) -> str:
    return f"p{n}_{model_id.lower()}"


def initial_spec(bundle: TaskBundle) -> PipelineSpec:
    return PipelineSpec(
        pipeline_id="p0_f0",
        data_view=DataView(),
        model_id=bundle.spec.f0_model,
        model_config={},
        seed=0,
        parent_id=None,
    )


def apply_action(action: Action, parent: PipelineSpec, n: int) -> PipelineSpec:
    """Produce the child spec from a single-field action."""
    spec = PipelineSpec(
        pipeline_id=make_pipeline_id(parent.model_id, n),
        data_view=copy.deepcopy(parent.data_view),
        model_id=parent.model_id,
        model_config=dict(parent.model_config),
        seed=parent.seed,
        parent_id=parent.pipeline_id,
    )
    if action.op == "SELECT_MODEL":
        spec.model_id = action.value
        spec.model_config = {}
        spec.pipeline_id = make_pipeline_id(action.value, n)
    elif action.op in ("SET_TRANSFORM", "SELECT_DATA"):
        setattr(spec.data_view, action.target_field, action.value)
    elif action.op == "SET_MODEL_OPTION":
        spec.model_config[action.target_field] = action.value
        spec.pipeline_id = make_pipeline_id(f"{spec.model_id}_{action.target_field}", n)
    return spec


def run_search(
    bundle: TaskBundle,
    cfg: dict[str, Any],
    run_dir: Path,
    seed: int = 0,
    ablation: str | None = None,
    controller: Controller | None = None,
    harness: Harness | None = None,
) -> dict[str, Any]:
    if ablation and ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation}; have {sorted(ABLATIONS)}")

    run_dir = Path(run_dir)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    h = harness or Harness(bundle, cfg, run_dir, seed=seed)
    ctrl = controller or Controller(cfg, run_dir)

    # f0 must run first so every delta_vs_f0 is defined
    f0 = initial_spec(bundle)
    f0.seed = seed
    h.f0_id = f0.pipeline_id
    fb0 = h.run_trial(f0, split="search")
    h.update_incumbent(f0, fb0)
    h.write_trace({"event": "f0", "pipeline": f0.to_dict(), "metrics": fb0.metrics})

    incumbent = f0
    no_improve = 0
    patience = cfg["budget"]["patience"]
    rounds = 0
    history: list[dict[str, Any]] = []

    while not h.exhausted() and no_improve < patience:
        rounds += 1
        stats = observe_mod.statistics(bundle, incumbent.data_view, "search")
        card = {} if ablation == "no_semantics" else observe_mod.semantic_card(bundle)
        figs = [] if ablation == "no_vision" else observe_mod.make_plots(
            bundle, incumbent.data_view, "search", run_dir / "figures", seed=seed)
        obs = Observation(
            task_id=bundle.spec.task_id,
            round_index=rounds - 1,
            dataset_card=card,
            statistics=stats,
            plot_paths=figs,
            archive_summary=[{"pipeline_id": e["pipeline"]["pipeline_id"],
                              "status": e["status"],
                              "normalized_mae": e["metrics"].get("normalized_mae"),
                              "model": e["pipeline"]["model_id"]} for e in h.archive[-6:]],
            incumbent={"pipeline_id": incumbent.pipeline_id,
                       "model_id": incumbent.model_id,
                       "data_view": incumbent.data_view.to_dict()},
            last_feedback=(h.last_feedback.to_dict() if h.last_feedback else None),
            budget=h.remaining(),
            allowed_ops=["SELECT_DATA", "SET_TRANSFORM", "SELECT_MODEL", "SET_MODEL_OPTION", "INSPECT", "STOP"],
            action_space={"models": list(_runnable()), "data_fields": sorted(DataView().__dict__.keys())},
        )

        h.spend_llm_call()
        t_llm = time.time()
        action = ctrl.propose(obs)
        h.write_trace({"event": "propose", "round": rounds, "action": action.to_dict(),
                       "llm_wall_s": round(time.time() - t_llm, 3),
                       "figures": [f.get("sha256") for f in figs]})

        if action.op == "STOP":
            h.write_trace({"event": "stop", "round": rounds, "reason": action.expected_effect})
            break
        if action.op == "INSPECT":
            # diagnostics only: does not consume a trial, re-loop
            h.write_trace({"event": "inspect", "round": rounds})
            continue

        checks = h.validate(action, incumbent)
        if not h.checks_ok(checks):
            log.info("round %d rejected: %s", rounds, checks)
            h.write_trace({"event": "rejected", "round": rounds, "checks": checks})
            no_improve += 1
            continue

        child = apply_action(action, incumbent, rounds)
        child.seed = seed
        fb = h.run_trial(child, split="search")
        accepted, delta = h.update_incumbent(child, fb)
        fb.delta_vs_f0 = _delta_f0(h, fb)
        h.write_trace({"event": "feedback", "round": rounds,
                       "pipeline": child.to_dict(), "status": fb.status,
                       "metrics": fb.metrics, "delta_vs_parent": delta,
                       "accepted": accepted, "error": fb.error})

        history.append({"round": rounds, "action": action.to_dict(),
                        "pipeline_id": child.pipeline_id, "status": fb.status,
                        "normalized_mae": fb.metrics.get("normalized_mae"),
                        "accepted": accepted})
        if accepted:
            incumbent = child
            no_improve = 0
        else:
            no_improve += 1
        if ablation == "one_shot" and rounds >= 1:
            break

    return {
        "task_id": bundle.spec.task_id,
        "seed": seed,
        "ablation": ablation,
        "incumbent": incumbent.to_dict(),
        "rounds": rounds,
        "used": h.used,
        "remaining": h.remaining(),
        "archive": h.archive,
        "history": history,
        "incumbent_metric": h._incumbent_metric(),
        "f0_metric": h.f0_metric(),
    }


def _delta_f0(h: Harness, fb) -> float | None:
    f0 = h.f0_metric()
    from . import evaluator
    return evaluator.improvement(f0, fb.metrics.get(h.eval_cfg["primary_metric"]))


def _runnable() -> list[str]:
    from .models.registry import runnable_ids
    return runnable_ids()


def finalize(
    bundle: TaskBundle,
    cfg: dict[str, Any],
    run_dir: Path,
    candidates: list[PipelineSpec],
    seed: int = 0,
    harness: Harness | None = None,
) -> dict[str, Any]:
    """Deterministic selection: confirm picks one, then test is scored once."""
    run_dir = Path(run_dir)
    h = harness or Harness(bundle, cfg, run_dir, seed=seed)
    h.f0_id = h.f0_id or "p0_f0"

    locked = candidates[:3]
    confirm_scores: list[dict[str, Any]] = []
    for spec in locked:
        fb = h.run_trial(spec, split="confirm")
        confirm_scores.append({"pipeline_id": spec.pipeline_id, "status": fb.status,
                               "normalized_mae": fb.metrics.get("normalized_mae"),
                               "error": fb.error})
    ok = [c for c in confirm_scores if c["normalized_mae"] is not None]
    if not ok:
        return {"status": "no_confirmable_candidate", "confirm": confirm_scores}
    best = min(ok, key=lambda c: c["normalized_mae"])
    winner = next(s for s in locked if s.pipeline_id == best["pipeline_id"])

    test_fb = h.evaluate_test(winner)
    out = {
        "status": "ok",
        "confirm": confirm_scores,
        "selected": winner.to_dict(),
        "test": {"status": test_fb.status, "metrics": test_fb.metrics,
                 "groups": test_fb.groups, "error": test_fb.error},
    }
    h.export(winner, run_dir)
    (run_dir / "finalize.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    h.write_trace({"event": "finalize", **{k: out[k] for k in ("status", "selected")}})
    return out
