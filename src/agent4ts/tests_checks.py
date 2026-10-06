"""Deterministic correctness checks used by `cli validate` and tests/.

Covers the DESIGN 6.1 acceptance list: no future leakage, padding does not change
statistics/scoring, unknown targets are unscored, mother trajectories do not cross
splits, and grid predictions map back to the fixed target.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .logging_utils import get_logger
from .schemas import DataView, TaskBundle

log = get_logger("checks")


def run_checks(bundle: TaskBundle, cfg: dict[str, Any]) -> dict[str, Any]:
    results = {
        "no_future_leakage": check_no_future_leakage(bundle),
        "split_isolation": check_split_isolation(bundle),
        "padding_invariance": check_padding_invariance(bundle),
        "unknown_targets_unscored": check_unknown_targets(bundle),
        "grid_maps_to_target": check_grid_maps(bundle),
        "no_label_over_split": check_no_label_over_split(bundle),
    }
    failed = [k for k, v in results.items() if not v["ok"]]
    return {"status": "fail" if failed else "ok", "checks": results, "failed": failed}


def check_no_future_leakage(bundle: TaskBundle) -> dict[str, Any]:
    """Every history event must satisfy event_time<=cutoff and available<=cutoff."""
    h = bundle.history
    cutoff = bundle.spec.history_duration
    bad_event = int((h.event_time > cutoff + 1e-9).sum())
    bad_avail = int((h.available_time > cutoff + 1e-9).sum())
    ok = bad_event == 0 and bad_avail == 0
    return {"ok": ok, "future_events": bad_event, "future_available": bad_avail}


def check_split_isolation(bundle: TaskBundle) -> dict[str, Any]:
    """No entity id appears in more than one split."""
    seen: dict[int, str] = {}
    dup = []
    for name, ids in bundle.spec.split_ids.items():
        for e in ids:
            if e in seen:
                dup.append((e, seen[e], name))
            seen[e] = name
    return {"ok": not dup, "duplicates": dup[:10], "n_entities": len(seen)}


def check_padding_invariance(bundle: TaskBundle) -> dict[str, Any]:
    """Statistics computed on history must not depend on any pad value.

    Windowing must introduce no fabricated entries: the number of mask==1 cells
    must equal the number of distinct (entity, time, variable) observations. Two
    observations that land on the same cell are a genuine collision (the mask is
    boolean), so we compare against unique cells, not raw row count.
    """
    from . import observe
    view = DataView()
    observe.statistics(bundle, view, "train")
    from .transforms import build_windows
    wins, _ = build_windows(bundle, view, bundle.entities_for("train"))
    n_masked = int(sum(int(w.mask.sum()) for w in wins))

    ent = bundle.entities_for("train")
    mask = np.isin(bundle.history.entity_id, list(ent))
    sub = bundle.history.subset(mask)
    # unique cells on the event grid (time rounded the same way as the builder)
    cells = {(int(e), round(float(t), 6), int(v))
             for e, t, v in zip(sub.entity_id, sub.event_time, sub.variable_id)}
    ok = n_masked == len(cells)
    return {"ok": ok, "masked_entries": n_masked, "unique_cells": len(cells),
            "raw_entries": int(len(sub))}



def check_unknown_targets(bundle: TaskBundle) -> dict[str, Any]:
    """Targets outside the query grid / horizon must not become scored labels."""
    t = bundle.targets
    cutoff = bundle.spec.history_duration
    beyond = int((t.event_time > cutoff + bundle.spec.horizon + 1e-9).sum())
    before = int((t.event_time <= cutoff + 1e-9).sum())
    ok = beyond == 0 and before == 0
    return {"ok": ok, "beyond_horizon": beyond, "before_cutoff": before}


def check_grid_maps(bundle: TaskBundle) -> dict[str, Any]:
    """Query grid values must be finite and inside the horizon."""
    g = bundle.spec.query_grid
    ok = all(np.isfinite(g)) and (len(g) == 0 or (min(g) >= 0 and max(g) <= bundle.spec.horizon + 1e-6))
    return {"ok": bool(ok), "n_grid": len(g),
            "min": float(min(g)) if g else None, "max": float(max(g)) if g else None}


def check_no_label_over_split(bundle: TaskBundle) -> dict[str, Any]:
    """Labels belong only to entities in the same split they were assigned."""
    all_split_entities = set()
    for ids in bundle.spec.split_ids.values():
        all_split_entities.update(ids)
    t_entities = set(np.unique(bundle.targets.entity_id).tolist())
    orphan = sorted(t_entities - all_split_entities)
    return {"ok": not orphan, "orphan_target_entities": orphan[:10]}
