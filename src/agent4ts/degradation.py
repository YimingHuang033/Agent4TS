"""Controlled history degradation for track B (`robust_improve`).

Each degradation clones the *history copy only*: same entity, same cutoff, same
query times and same ground-truth labels. Derived versions carry a tag so they
follow their mother trajectory into the same split. Applied one at a time, never
in full combination (DESIGN 2.3).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .logging_utils import get_logger
from .schemas import EventTable, TaskBundle

log = get_logger("degradation")

KINDS = ("drop_random", "drop_span", "subsample_vars", "add_noise")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def drop_random(history: EventTable, ratio: float, seed: int = 0) -> EventTable:
    if ratio <= 0:
        return history
    rng = _rng(seed)
    keep = rng.random(len(history)) >= ratio
    return history.subset(keep)


def drop_span(history: EventTable, fraction: float, seed: int = 0) -> EventTable:
    """Delete one contiguous time interval covering `fraction` of history span."""
    if fraction <= 0:
        return history
    rng = _rng(seed)
    span = history.event_time.max() - history.event_time.min()
    width = span * fraction
    start = history.event_time.min() + rng.random() * (span - width)
    inside = (history.event_time >= start) & (history.event_time <= start + width)
    return history.subset(~inside)


def subsample_vars(history: EventTable, keep_fraction: float, seed: int = 0) -> EventTable:
    """Keep observations for a random subset of auxiliary variables.

    Target variables are never dropped here so the task stays well-posed.
    """
    rng = _rng(seed)
    variables = np.unique(history.variable_id)
    if keep_fraction >= 1.0:
        return history
    n_keep = max(1, int(round(len(variables) * keep_fraction)))
    kept = set(rng.choice(variables, size=n_keep, replace=False).tolist())
    # targets are identified by the caller normally; here we keep the first var
    kept.add(int(variables[0]))
    mask = np.isin(history.variable_id, list(kept))
    return history.subset(mask)


def add_noise(history: EventTable, std_ratio: float, train_std: dict[int, float] | None,
              seed: int = 0) -> EventTable:
    """Gaussian noise with std = ratio * per-variable train std."""
    if std_ratio <= 0:
        return history
    rng = _rng(seed)
    noise = np.zeros(len(history))
    if train_std:
        for vid, s in train_std.items():
            m = history.variable_id == vid
            noise[m] = rng.normal(0.0, std_ratio * s, size=int(m.sum()))
    else:
        # fall back to per-variable empirical std in the history itself
        for vid in np.unique(history.variable_id):
            m = history.variable_id == vid
            s = float(np.std(history.value[m])) if m.sum() > 1 else 0.0
            noise[m] = rng.normal(0.0, std_ratio * s, size=int(m.sum()))
    return EventTable(
        entity_id=history.entity_id,
        event_time=history.event_time,
        available_time=history.available_time,
        variable_id=history.variable_id,
        value=history.value + noise,
    )


def apply_degradation(bundle: TaskBundle, spec: dict[str, Any], seed: int = 0) -> TaskBundle:
    """Return a new bundle with degraded history; targets unchanged.

    spec form: {"kind": "drop_random", "ratio": 0.5, "tag": "d020"}
    """
    kind = spec["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown degradation kind {kind}; have {KINDS}")
    hist = bundle.history
    if kind == "drop_random":
        new = drop_random(hist, spec["ratio"], seed)
    elif kind == "drop_span":
        new = drop_span(hist, spec["fraction"], seed)
    elif kind == "subsample_vars":
        new = subsample_vars(hist, spec["keep_fraction"], seed)
    elif kind == "add_noise":
        new = add_noise(hist, spec["std_ratio"], None, seed)
    else:  # pragma: no cover
        raise AssertionError(kind)

    tag = spec.get("tag", kind)
    import copy
    out = copy.copy(bundle)
    out.history = new
    out.spec = copy.copy(bundle.spec)
    out.spec.degradation = {**spec, "applied_tag": tag}
    out.spec.task_id = f"{bundle.spec.task_id}_{tag}"
    out.spec.dataset_version = f"{bundle.spec.dataset_version}__{tag}"
    log.info("degradation %s applied: %d -> %d events", tag, len(hist), len(new))
    return out


def default_grid() -> list[dict[str, Any]]:
    """The degradation settings named in DESIGN 2.3, applied one at a time."""
    return [
        {"kind": "drop_random", "ratio": 0.2, "tag": "dr020"},
        {"kind": "drop_random", "ratio": 0.5, "tag": "dr050"},
        {"kind": "drop_span", "fraction": 0.1, "tag": "ds010"},
        {"kind": "drop_span", "fraction": 0.3, "tag": "ds030"},
        {"kind": "subsample_vars", "keep_fraction": 0.5, "tag": "sv050"},
        {"kind": "subsample_vars", "keep_fraction": 0.25, "tag": "sv025"},
        {"kind": "add_noise", "std_ratio": 0.1, "tag": "no010"},
        {"kind": "add_noise", "std_ratio": 0.3, "tag": "no030"},
    ]
