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

    stale_age = eval_cfg.get("stale_age_hours", 3.0)
    long_gap = eval_cfg.get("long_gap_hours", 6.0)

    all_nmae, all_nmse = [], []
    per_var: dict[int, list[tuple[float, float]]] = {}
    groups: dict[str, list[tuple[float, float]]] = {
        "all": [], "long_gap": [], "stale": [], "fresh": [], "sparse": [], "dense": []}

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
            all_nmae.append(nmae_q); all_nmse.append(nmse_q)
            per_var.setdefault(vid, []).append((nmae_q, nmse_q))
            groups["all"].append((nmae_q, nmse_q))

            # per-query diagnostics derived from the window's real observations
            pi = pos.get(vid)
            gap, age = _query_diagnostics(lw, pi, q)
            if gap is not None and gap >= long_gap:
                groups["long_gap"].append((nmae_q, nmse_q))
            if age is not None and age >= stale_age:
                groups["stale"].append((nmae_q, nmse_q))
            else:
                groups["fresh"].append((nmae_q, nmse_q))
            dens = _density(lw, pi)
            if dens == "sparse":
                groups["sparse"].append((nmae_q, nmse_q))
            elif dens == "dense":
                groups["dense"].append((nmae_q, nmse_q))

    def _agg(pairs: list[tuple[float, float]]) -> dict[str, Any]:
        if not pairs:
            return {"nmae": None, "nmse": None, "n": 0}
        arr = np.asarray(pairs)
        return {"nmae": float(arr[:, 0].mean()), "nmse": float(arr[:, 1].mean()), "n": int(arr.shape[0])}

    per_variable = {
        bundle.var_name(vid): _agg(vals) for vid, vals in sorted(per_var.items())
    }
    overall = _agg(groups["all"])
    return {
        "metrics": {
            "normalized_mae": overall["nmae"],
            "normalized_mse": overall["nmse"],
            "n_valid_targets": n_valid,
            "coverage": (n_valid / n_total) if n_total else None,
        },
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
