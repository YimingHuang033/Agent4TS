"""Restricted data transforms.

The agent may only pick from these registered modules; it never writes free
code. Each transform is a pure function over the event table + query grid and
returns padded tensors *plus a boolean mask* so padding is never mistaken for a
real observation. Preprocessing is fit on the train split only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import DataView, EventTable, TaskBundle


@dataclass
class Window:
    """One padded model input + its masks."""
    x: np.ndarray          # (n_vars, T) values
    mask: np.ndarray       # (n_vars, T) 1 = real observation
    age: np.ndarray        # (n_vars, T) hours since last real obs (0 at obs)
    times: np.ndarray      # (T,) relative hours on the grid
    variable_ids: np.ndarray  # (n_vars,)
    entity_id: int
    cutoff: float


@dataclass
class LabeledWindow:
    window: Window
    query_times: np.ndarray      # (Q,)
    query_var_ids: np.ndarray    # (Q,)
    y_true: np.ndarray           # (Q,)   NaN where unscored
    y_scored: np.ndarray         # (Q,) bool


# ------------------------------------------------------------------ normalizer

class Normalizer:
    """z-score / min-max fitted on train entities only."""

    def __init__(self, mode: str = "zscore"):
        self.mode = mode
        self.mean: dict[int, float] = {}
        self.std: dict[int, float] = {}
        self.lo: dict[int, float] = {}
        self.hi: dict[int, float] = {}

    @classmethod
    def fit(cls, history: EventTable, entity_ids: np.ndarray, mode: str = "zscore") -> "Normalizer":
        nz = cls(mode)
        mask = np.isin(history.entity_id, entity_ids)
        for vid in np.unique(history.variable_id[mask]):
            v = history.value[mask & (history.variable_id == vid)]
            if v.size == 0:
                continue
            nz.mean[int(vid)] = float(np.mean(v))
            s = float(np.std(v))
            nz.std[int(vid)] = s if s > 1e-8 else 1.0
            nz.lo[int(vid)] = float(np.min(v))
            nz.hi[int(vid)] = float(np.max(v))
        return nz

    def transform(self, vid: int, value: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return value
        if self.mode == "minmax":
            lo, hi = self.lo.get(vid, 0.0), self.hi.get(vid, 1.0)
            rng = hi - lo
            return (value - lo) / (rng if abs(rng) > 1e-8 else 1.0)
        return (value - self.mean.get(vid, 0.0)) / self.std.get(vid, 1.0)

    def inverse(self, vid: int, value: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return value
        if self.mode == "minmax":
            lo, hi = self.lo.get(vid, 0.0), self.hi.get(vid, 1.0)
            return value * (hi - lo) + lo
        return value * self.std.get(vid, 1.0) + self.mean.get(vid, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "mean": self.mean, "std": self.std,
                "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Normalizer":
        nz = cls(d.get("mode", "zscore"))
        nz.mean = {int(k): v for k, v in d.get("mean", {}).items()}
        nz.std = {int(k): v for k, v in d.get("std", {}).items()}
        nz.lo = {int(k): v for k, v in d.get("lo", {}).items()}
        nz.hi = {int(k): v for k, v in d.get("hi", {}).items()}
        return nz


# ------------------------------------------------------------------ view build

def build_windows(
    bundle: TaskBundle,
    view: DataView,
    entity_ids: np.ndarray,
    norm: Normalizer | None = None,
) -> tuple[list[Window], list[np.ndarray]]:
    """Build padded windows for `entity_ids`. Returns (windows, raw_times).

    Two stages so training windows can be assembled without labels; the labeled
    variant below attaches them.
    """
    windows: list[Window] = []
    history = bundle.history
    cutoff = bundle.spec.history_duration
    keep_vars = _variable_selection(bundle, view)
    keep_var_ids = np.asarray(sorted(keep_vars), dtype=np.int64)

    # effective history span
    hist_start = cutoff - cutoff * view.history_fraction

    for eid in entity_ids.tolist():
        m = (history.entity_id == eid) & (history.event_time >= hist_start)
        if view.variable_subset == "targets_only":
            m &= np.isin(history.variable_id, [bundle.var_index(v) for v in bundle.spec.target_variables])
        sub = history.subset(m)
        win = _events_to_window(
            sub, keep_var_ids, cutoff, view, norm, bundle_anchors=bundle.entity_anchors,
        )
        if win is not None:
            windows.append(win)
    return windows, [w.times for w in windows]


def build_labeled_windows(
    bundle: TaskBundle, view: DataView, split: str, norm: Normalizer | None = None,
) -> list[LabeledWindow]:
    """Windows for one split with bucketed/point labels joined from targets."""
    entity_ids = bundle.entities_for(split)
    windows, _ = build_windows(bundle, view, entity_ids, norm)
    by_entity = {w.entity_id: w for w in windows}
    out: list[LabeledWindow] = []

    targets = bundle.targets
    tg_by_entity: dict[int, EventTable] = {}
    tm = np.isin(targets.entity_id, entity_ids)
    tsub = targets.subset(tm)
    for eid in np.unique(tsub.entity_id):
        tg_by_entity[int(eid)] = tsub.subset(tsub.entity_id == eid)

    for eid, win in by_entity.items():
        tg = tg_by_entity.get(eid)
        if tg is None or len(tg) == 0:
            continue
        win_cut = bundle.spec.history_duration
        qtimes = tg.event_time - win_cut          # relative to cutoff
        qvars = tg.variable_id.astype(np.int64)
        ytrue = tg.value.astype(np.float64)

        # collapse duplicates on (query_time, var) via mean = bucketed-mean label
        keys: dict[tuple[int, int], list[float]] = {}
        for t, v, y in zip(qtimes, qvars, ytrue):
            keys.setdefault((int(round(t * 1e6)), int(v)), []).append(float(y))
        qt, qv, yy = [], [], []
        for (tk, v), vals in keys.items():
            qt.append(tk / 1e6); qv.append(v); yy.append(float(np.mean(vals)))
        out.append(LabeledWindow(
            window=win,
            query_times=np.asarray(qt, dtype=np.float64),
            query_var_ids=np.asarray(qv, dtype=np.int64),
            y_true=np.asarray(yy, dtype=np.float64),
            y_scored=np.ones(len(yy), dtype=bool),
        ))
    return out


def _variable_selection(bundle: TaskBundle, view: DataView) -> set[int]:
    if view.variable_subset == "targets_only":
        return {bundle.var_index(v) for v in bundle.spec.target_variables}
    return set(range(len(bundle.variable_names)))


def _events_to_window(
    sub: EventTable,
    keep_var_ids: np.ndarray,
    cutoff: float,
    view: DataView,
    norm: Normalizer | None,
    bundle_anchors: dict[int, float],
) -> Window | None:
    if len(sub) == 0:
        return None
    eid = int(sub.entity_id[0])

    if view.grid_mode == "grid" and view.grid_width > 0:
        edges = np.arange(cutoff - cutoff * view.history_fraction, cutoff + 1e-9, view.grid_width)
        times = edges[:-1]
        T = len(times)
        x = np.full((len(keep_var_ids), T), np.nan)
        mask = np.zeros((len(keep_var_ids), T), dtype=np.int64)
        vid_pos = {int(v): i for i, v in enumerate(keep_var_ids)}
        # nearest bucket by floor((t - t0)/w)
        for t, v, val in zip(sub.event_time, sub.variable_id, sub.value):
            vi = vid_pos.get(int(v))
            if vi is None:
                continue
            b = int((t - times[0]) // view.grid_width)
            if 0 <= b < T:
                x[vi, b] = val if np.isnan(x[vi, b]) else 0.5 * (x[vi, b] + val)
                mask[vi, b] = 1
    else:
        # event direct read: union of observation times, per-var NaN elsewhere
        times = np.unique(sub.event_time)
        T = len(times)
        x = np.full((len(keep_var_ids), T), np.nan)
        mask = np.zeros((len(keep_var_ids), T), dtype=np.int64)
        vid_pos = {int(v): i for i, v in enumerate(keep_var_ids)}
        tpos = {round(float(t), 6): i for i, t in enumerate(times)}
        for t, v, val in zip(sub.event_time, sub.variable_id, sub.value):
            vi = vid_pos.get(int(v));  ti = tpos.get(round(float(t), 6))
            if vi is None or ti is None:
                continue
            x[vi, ti] = val; mask[vi, ti] = 1

    age = _compute_age(x, mask, times, view.forward_limit)

    if view.fill in ("mean", "ffill"):
        x = _fill(x, mask, times, view)

    if norm is not None:
        for i, vid in enumerate(keep_var_ids):
            real = mask[i].astype(bool)
            if real.any():
                x[i, real] = norm.transform(int(vid), x[i, real])
                # filled values are transformed with the same train stats
                fv = ~real
                if fv.any():
                    x[i, fv] = norm.transform(int(vid), x[i, fv])

    return Window(x=x, mask=mask, age=age, times=times,
                  variable_ids=keep_var_ids, entity_id=eid, cutoff=cutoff)


def _compute_age(x: np.ndarray, mask: np.ndarray, times: np.ndarray, forward_limit: float) -> np.ndarray:
    nv, T = x.shape
    age = np.full((nv, T), np.nan)
    for i in range(nv):
        last = None
        for j in range(T):
            if mask[i, j]:
                last = times[j]
                age[i, j] = 0.0
            elif last is not None:
                a = times[j] - last
                age[i, j] = a
    return age


def _fill(x: np.ndarray, mask: np.ndarray, times: np.ndarray, view: DataView) -> np.ndarray:
    out = x.copy()
    nv, T = out.shape
    for i in range(nv):
        real = mask[i].astype(bool)
        if view.fill == "mean":
            fill_val = float(np.nanmean(out[i, real])) if real.any() else 0.0
            out[i, ~real] = fill_val
        else:  # forward hold with a validity limit
            last = np.nan
            last_t = np.nan
            for j in range(T):
                if real[j]:
                    last = out[i, j]; last_t = times[j]
                else:
                    fresh = not np.isnan(last) and (view.forward_limit <= 0 or (times[j] - last_t) <= view.forward_limit)
                    out[i, j] = last if fresh else np.nan
            if np.isnan(out[i]).any():
                fill_val = float(np.nanmean(out[i])) if not np.isnan(out[i]).all() else 0.0
                out[i, np.isnan(out[i])] = fill_val
    return out


# ------------------------------------------------------------------ validation

def compatible(bundle: TaskBundle, view: DataView) -> tuple[bool, str]:
    """Compatibility matrix: reject combinations that make no sense."""
    if view.history_fraction not in (0.5, 1.0):
        return False, f"history_fraction must be in {{0.5, 1.0}}, got {view.history_fraction}"
    if view.grid_mode not in ("events", "grid"):
        return False, f"grid_mode must be events|grid, got {view.grid_mode}"
    if view.grid_mode == "grid" and view.grid_width <= 0:
        return False, "grid width must be positive"
    if view.fill not in ("mean", "ffill"):
        return False, f"fill must be mean|ffill, got {view.fill}"
    if view.normalize not in ("zscore", "minmax", "none"):
        return False, f"normalize must be zscore|minmax|none, got {view.normalize}"
    if view.forward_limit < 0:
        return False, "forward_limit must be >= 0"
    # interval-rate transform only for audited RepoHealth (not registered yet)
    if bundle.spec.backend != "timeimm" and view.variable_subset == "targets_only" and bundle.spec.track.value == "native_irr":
        return True, ""
    return True, ""
