"""Observation for the agent: statistics + real PNG figures + semantic card.

Every figure is saved as an actual PNG on disk and its path AND bytes hash are
returned, so the multimodal controller really receives an image (DESIGN 5.1),
not a bare path. Test/confirm splits are never plotted, and no figure uses
information from after the entity's cutoff.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import get_logger
from .schemas import DataView, TaskBundle
from .transforms import build_windows

log = get_logger("observe")

# small, deterministic palette so plots are reproducible
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def statistics(bundle: TaskBundle, view: DataView, split: str) -> dict[str, Any]:
    """Numeric summary the agent sees: coverage, gap and sampling stats."""
    hist = bundle.history
    ents = set(bundle.entities_for(split).tolist())
    mask = np.isin(hist.entity_id, list(ents))
    sub = hist.subset(mask)
    if len(sub) == 0:
        return {"n_events": 0, "variables": {}, "note": "empty split"}

    cutoff = bundle.spec.history_duration
    per_var: dict[str, Any] = {}
    for vid in np.unique(sub.variable_id):
        m = sub.variable_id == vid
        t = sub.event_time[m]
        v = sub.value[m]
        gaps = np.diff(np.sort(t)) if t.size > 1 else np.array([])
        per_var[bundle.var_name(int(vid))] = {
            "n": int(m.sum()),
            "unit": bundle.card.unit(bundle.var_name(int(vid))),
            "obs_per_entity": float(m.sum() / max(1, len(ents))),
            "coverage_frac": float(m.sum() / max(1, len(ents) * cutoff)),
            "median_gap_h": float(np.median(gaps)) if gaps.size else None,
            "max_gap_h": float(gaps.max()) if gaps.size else None,
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
        }
    age_all = cutoff - sub.event_time
    return {
        "n_events": int(len(sub)),
        "n_entities": len(ents),
        "cutoff_hours": cutoff,
        "horizon_hours": bundle.spec.horizon,
        "history_duration": bundle.spec.history_duration,
        "obs_age_h": {"mean": float(age_all.mean()), "p90": float(np.percentile(age_all, 90))},
        "variables": per_var,
        "target_variables": bundle.spec.target_variables,
        "query_policy": bundle.spec.query_policy.value,
        "degradation": bundle.spec.degradation,
    }


def semantic_card(bundle: TaskBundle) -> dict[str, Any]:
    c = bundle.card
    return {
        "dataset_version": c.dataset_version,
        "backend": c.backend,
        "time_semantics": c.time_semantics,
        "variables": c.variables,
        "unknown": c.unknown,
        "provenance": c.provenance,
    }


def make_plots(
    bundle: TaskBundle,
    view: DataView,
    split: str,
    out_dir: Path,
    max_figs: int = 4,
    max_vars_per_fig: int = 6,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Render up to `max_figs` PNGs on the real time axis. Returns path+hash."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    entity_ids = bundle.entities_for(split)
    if entity_ids.size == 0:
        return []
    rng = np.random.default_rng(seed)
    sample = entity_ids[rng.permutation(entity_ids.size)[:max_figs]]

    figures: list[dict[str, Any]] = []
    for k, eid in enumerate(sample.tolist()):
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
        sub = bundle.history.subset(bundle.history.entity_id == eid)
        if len(sub) == 0:
            plt.close(fig); continue
        variables = np.unique(sub.variable_id)[:max_vars_per_fig]
        for i, vid in enumerate(variables):
            m = sub.variable_id == vid
            t = sub.event_time[m]
            v = sub.value[m]
            # scatter, not a line: long gaps must not read as a continuous curve
            ax.scatter(t, v, s=10, color=_COLORS[i % len(_COLORS)], alpha=0.8,
                       label=f"{bundle.var_name(int(vid))} [{bundle.card.unit(bundle.var_name(int(vid)))}]")
        ax.axvline(bundle.spec.history_duration, color="k", ls="--", lw=1, label="cutoff")
        ax.set_xlabel("relative time (h)")
        ax.set_title(f"entity {eid} — {split} history (scatter=real obs)")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(alpha=0.25)
        path = out_dir / f"e{k}_history_eid{eid}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        figures.append({"path": str(path), "sha256": _hash(path), "kind": "history",
                        "entity_id": eid, "variables": [bundle.var_name(int(v)) for v in variables]})

    # one coverage figure across entities in the split
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    sub = bundle.history.subset(np.isin(bundle.history.entity_id, entity_ids))
    for i, name in enumerate(bundle.variable_names[:max_vars_per_fig]):
        vid = bundle.var_index(name)
        m = sub.variable_id == vid
        if not m.any():
            continue
        counts = np.array([int(((sub.entity_id == e) & (sub.variable_id == vid)).sum())
                           for e in entity_ids.tolist()])
        ax.plot(np.arange(len(counts)), counts, color=_COLORS[i % len(_COLORS)], label=name)
    ax.set_xlabel("entity index"); ax.set_ylabel("real observations")
    ax.set_title(f"{split} coverage per entity")
    ax.legend(fontsize=6); ax.grid(alpha=0.25)
    path = out_dir / "coverage.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    figures.append({"path": str(path), "sha256": _hash(path), "kind": "coverage",
                    "variables": bundle.variable_names[:max_vars_per_fig]})
    return figures


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_png_bytes(fig: dict[str, Any]) -> bytes:
    """Read the PNG so a multimodal controller can attach the real image."""
    return Path(fig["path"]).read_bytes()
