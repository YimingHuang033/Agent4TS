"""Evaluator: deterministic scoring, groups and comparison.

Primary metric is normalized MAE (train-side standardization, per DESIGN 5).
Predictions and truth are compared in original units; the normalizer supplied by
the harness converts both to the train-standardized space. Missing truth is never
scored, and an empty group returns null with n=0 (never a fake zero).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .logging_utils import get_logger
from .schemas import TaskBundle
from .transforms import LabeledWindow, Normalizer

log = get_logger("evaluator")


def score_windows(
    labeled: list[LabeledWindow],
    preds: list[np.ndarray],
    bundle: TaskBundle,
    norm: Normalizer,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute metrics + robustness groups for one split."""
    if len(labeled) != len(preds):
        raise ValueError(f"labeled/preds length mismatch: {len(labeled)} vs {len(preds)}")

    diagnostics, thresholds = robustness_diagnostics(bundle)
    stale_age = thresholds["age"]
    long_gap = thresholds["gap"]

    entity_variable_errors: dict[tuple, list] = {}
    raw_per_var: dict[int, list] = {}
    per_var: dict[int, list[tuple[float, float]]] = {}
    groups: dict[str, list[tuple[float, float]]] = {
        "all": [], "long_gap": [], "stale": [], "fresh": [], "sparse": [], "dense": [], "async": [], "density_shift": []}

    n_valid = 0
    n_total = 0
    for lw, pred in zip(labeled, preds):
        pos = {int(v): i for i, v in enumerate(lw.window.variable_ids)}
        for q in range(len(lw.query_times)):
            n_total += 1
            if not lw.y_scored[q] or not np.isfinite(lw.y_true[q]) or not np.isfinite(pred[q]):
                continue
            vid = int(lw.query_var_ids[q])
            y = lw.y_true[q]
            p = pred[q]
            # normalized error: divide the raw residual by the train std, so MAE
            # stays comparable across variables of different scale
            s = norm.std.get(vid, 1.0)
            n_err = (p - y) / s
            nmae_q = abs(n_err)
            nmse_q = n_err ** 2
            n_valid += 1
            entity_variable_errors.setdefault((lw.window.entity_id, vid), []).append((nmae_q, nmse_q))
            raw_per_var.setdefault(vid, []).append((abs(p - y), (p - y) ** 2))
            per_var.setdefault(vid, []).append((lw.window.entity_id, vid, nmae_q, nmse_q))
            groups["all"].append((lw.window.entity_id, vid, nmae_q, nmse_q))

            # per-query diagnostics derived from the window's real observations
            pi = pos.get(vid)
            diag = diagnostics.get((lw.window.entity_id, vid), {})
            gap, age = diag.get("gap"), diag.get("age")
            if diag.get("async", 0) > thresholds["async"]:
                groups["async"].append((lw.window.entity_id, vid, nmae_q, nmse_q))
            if diag.get("density_shift", 0) > thresholds["density_shift"]:
                groups["density_shift"].append((lw.window.entity_id, vid, nmae_q, nmse_q))
            if gap is not None and gap >= long_gap:
                groups["long_gap"].append((lw.window.entity_id, vid, nmae_q, nmse_q))
            if age is not None and age >= stale_age:
                groups["stale"].append((lw.window.entity_id, vid, nmae_q, nmse_q))
            else:
                groups["fresh"].append((lw.window.entity_id, vid, nmae_q, nmse_q))
            density = diag.get("density")
            dens = ("sparse" if density <= thresholds["density_low"] else "dense" if density >= thresholds["density_high"] else None) if density is not None else None
            if dens == "sparse":
                groups["sparse"].append((lw.window.entity_id, vid, nmae_q, nmse_q))
            elif dens == "dense":
                groups["dense"].append((lw.window.entity_id, vid, nmae_q, nmse_q))

    def _agg(pairs: list[tuple[float, float]]) -> dict[str, Any]:
        if not pairs:
            return {"nmae": None, "nmse": None, "n": 0}
        arr = np.asarray(pairs)
        n = len(arr)
        if arr.shape[1] == 4:
            grouped = {}
            for eid, vid, mae, mse in arr:
                grouped.setdefault((eid, vid), []).append((mae, mse))
            arr = np.asarray([np.mean(v, axis=0) for v in grouped.values()])
        return {"nmae": float(arr[:, 0].mean()), "nmse": float(arr[:, 1].mean()), "n": n}

    per_variable = {
        bundle.var_name(vid): _agg(vals) for vid, vals in sorted(per_var.items())
    }
    # Give every entity-variable pair equal weight, independently of the
    # number of observed targets. This is the DESIGN.md macro protocol.
    pair_means = [np.asarray(values).mean(axis=0) for values in entity_variable_errors.values()]
    overall = _agg(pair_means)
    for vid, values in raw_per_var.items():
        raw = np.asarray(values)
        per_variable[bundle.var_name(vid)].update(
            mae=float(raw[:, 0].mean()), mse=float(raw[:, 1].mean()))
    return {
        "metrics": {
            "normalized_mae": overall["nmae"],
            "normalized_mse": overall["nmse"],
            "n_valid_targets": n_valid,
            "n_entity_variable_pairs": len(pair_means),
            "coverage": (n_valid / n_total) if n_total else None,
        },
        "group_thresholds": thresholds,
        "groups": {k: _agg(v) for k, v in groups.items()},
        "per_variable": per_variable,
        "support_counts": {"n_total": n_total, "n_valid": n_valid,
                           "n_long_gap": groups["long_gap"].__len__(),
                           "n_stale": groups["stale"].__len__()},
    }


def _query_diagnostics(lw: LabeledWindow, var_pos: int | None, q: int) -> tuple[float | None, float | None]:
    """Max gap and last-observation age relevant to this query."""
    if var_pos is None:
        return None, None
    real = lw.window.mask[var_pos].astype(bool)
    if not real.any():
        return None, None
    ts = lw.window.times[real]
    qt = lw.query_times[q] + lw.window.cutoff
    prior = ts[ts <= lw.window.cutoff]
    if prior.size == 0:
        return None, None
    last_t = prior[-1]
    age = lw.window.cutoff - last_t
    gaps = np.diff(prior)
    max_gap = float(gaps.max()) if gaps.size else 0.0
    return max_gap, float(age)


def _density(lw: LabeledWindow, var_pos: int | None) -> str | None:
    if var_pos is None:
        return None
    real = lw.window.mask[var_pos].astype(bool)
    n = int(real.sum())
    if lw.window.times.size == 0:
        return None
    rate = n / lw.window.times.size
    if rate < 0.05:
        return "sparse"
    if rate > 0.3:
        return "dense"
    return None


def improvement(parent: float | None, child: float | None) -> float | None:
    """Relative improvement; zero denominator -> null, never a fabricated number."""
    if parent is None or child is None or parent == 0:
        return None
    return (parent - child) / parent


def robustness_diagnostics(bundle):
    """Use original history, invariant to candidate transforms; fit train quantiles only."""
    history = bundle.history
    diagnostics = {}
    duration = bundle.spec.history_duration
    order = np.argsort(history.entity_id, kind="stable")
    ids, starts = np.unique(history.entity_id[order], return_index=True)
    pieces = np.split(order, starts[1:])
    entity_rows = dict(zip(ids, pieces))
    for eid, cutoff in bundle.entity_anchors.items():
        ix = entity_rows.get(eid, np.array([], dtype=int))
        times = history.event_time[ix]
        vids = history.variable_id[ix]
        valid = ((times <= cutoff) & (history.available_time[ix] <= cutoff) & (times >= cutoff - duration))
        ages = []
        rows = []
        for vid in range(len(bundle.variable_names)):
            ts = np.unique(times[valid & (vids == vid)])
            if not len(ts):
                continue
            age = float(cutoff - ts[-1]); ages.append(age)
            recent = int((ts > cutoff - duration / 2).sum())
            earlier = len(ts) - recent
            row = {"age": age, "gap": float(np.diff(np.r_[cutoff-duration, ts, cutoff]).max()),
                   "density": len(ts) / duration,
                   "density_shift": abs(float(np.log((recent + 1) / (earlier + 1))))}
            diagnostics[eid, vid] = row; rows.append(row)
        for row in rows:
            row["async"] = max(ages) - min(ages)
    train = set(bundle.entities_for("train"))
    fit = [v for (eid, _), v in diagnostics.items() if eid in train]
    if not fit:
        raise ValueError("No training observations to fit robustness thresholds")
    thresholds = {key: float(np.quantile([r[key] for r in fit], .75))
                  for key in ("age", "gap", "async", "density_shift")}
    thresholds.update(density_low=float(np.quantile([r['density'] for r in fit], .25)),
                      density_high=float(np.quantile([r['density'] for r in fit], .75)))
    return diagnostics, thresholds
