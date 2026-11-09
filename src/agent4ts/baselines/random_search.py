"""Baselines: random search over the same action space, matching budget/seed."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..harness import Harness
from ..logging_utils import get_logger
from ..models.registry import runnable_ids
from ..schemas import DataView, PipelineSpec, TaskBundle
from ..search import initial_spec, apply_action, TRANSFORM_SPACE
from ..schemas import Action
from ..models.registry import CATALOG

log = get_logger("baselines.random")

DATA_CHOICES = {
    "history_fraction": [0.5, 1.0],
    "grid_mode": ["events", "grid"],
    "grid_width": [1.0, 2.0],
    "fill": ["mean", "ffill"],
    "forward_limit": [0.0, 3.0, 6.0],
    "variable_subset": ["all", "targets_only"],
    "normalize": ["zscore", "minmax", "none"],
    "age_feature": [False, True],
}


def run_random_search(
    bundle: TaskBundle,
    cfg: dict[str, Any],
    run_dir: Path,
    seed: int = 0,
    harness: Harness | None = None,
) -> dict[str, Any]:
    """Same budget, same action space, no LLM. Uses a fixed RNG for reproducibility."""
    run_dir = Path(run_dir)
    h = harness or Harness(bundle, cfg, run_dir, seed=seed)
    rng = random.Random(seed)

    f0 = initial_spec(bundle); f0.seed = seed
    h.f0_id = f0.pipeline_id
    fb0 = h.run_trial(f0, split="search")
    h.update_incumbent(f0, fb0)

    incumbent = f0
    no_improve = 0
    patience = cfg["budget"]["patience"]
    rounds = 0
    while not h.exhausted() and no_improve < patience:
        rounds += 1
        if rounds > cfg['budget']['trials'] * 100:
            break
        choices = [Action(op='SELECT_MODEL', value=m) for m in runnable_ids()]
        choices += [Action(op='SET_TRANSFORM', target_field=field, value=value)
                    for field, values in TRANSFORM_SPACE.items() for value in values]
        choices += [Action(op='SET_MODEL_OPTION', target_field=field, value=value)
                    for field, values in CATALOG[incumbent.model_id]['options'].items() for value in values]
        action = rng.choice(choices)
        if not h.checks_ok(h.validate(action, incumbent)):
            continue
        spec = apply_action(action, incumbent, rounds)
        spec.seed = seed
        if h.prior_trial(spec) is not None:
            continue
        fb = h.run_trial(spec, split="search")
        accepted, _ = h.update_incumbent(spec, fb)
        h.write_trace({"event": "random", "round": rounds, "pipeline": spec.to_dict(),
                       "status": fb.status, "metrics": fb.metrics, "accepted": accepted})
        if accepted:
            incumbent = spec; no_improve = 0
        else:
            no_improve += 1

    out = {"task_id": bundle.spec.task_id, "seed": seed, "rounds": rounds,
           "incumbent": incumbent.to_dict(), "archive": h.archive,
           "incumbent_metric": h._incumbent_metric(), "f0_metric": h.f0_metric(),
           "used": h.used}
    (run_dir / "random_search.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
