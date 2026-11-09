"""Physiome-ODE backend (mechanistic simulation, track A + B).

Builds a synthetic irregular series from an ODE but keeps a *single latent
trajectory* per entity: history versions are derived by masking the same latent
trajectory, and the future target is identical across versions. The full future
latent trajectory is only ever handed to the evaluator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..logging_utils import get_logger
from ..schemas import DatasetCard, EventTable, QueryPolicy, TaskBundle, TaskSpec, Track

log = get_logger("data.physiome")

# The three models named in DESIGN 2.2, expressed as explicit ODE right-hand
# sides so the identifiers in config map to real dynamics. Time scaling is
# declared per model so event_time is always in hours.
MODELS: dict[str, dict] = {
    "dupont_1991a": {
        "vars": ["x", "y"],
        "params": {"a": 0.04, "b": 1.0e6, "c": 0.04, "v": 1.5, "k": 100.0},
        "rhs": lambda s, p, t: np.array([
            p["v"] / (1 + p["k"] * s[1] ** 2) - 0.0 * s[0],
            p["a"] * (p["b"] / (1 + s[0] ** 2) - s[1]),
        ]),
        "time_scale_hours": 60.0,
        "init": [0.5, 0.5],
    },
    "borghans_dupont_goldbeter_1997a": {
        "vars": ["x", "y", "z"],
        "params": {"v0": 0.4, "vm2": 0.6, "vm3": 2.0, "k2": 0.1, "kr": 1.0,
                   "ka": 0.6, "ki": 0.4, "k3": 1.0, "n": 4.0},
        "rhs": lambda s, p, t: np.array([
            p["v0"] + p["vm2"] * s[2] ** 2 / (p["kr"] ** 2 + s[2] ** 2)
            - p["k3"] * s[0],
            p["vm2"] * s[2] ** 2 / (p["kr"] ** 2 + s[2] ** 2) - p["ka"] * s[1],
            p["vm3"] * s[1] ** p["n"] / (p["ki"] ** p["n"] + s[1] ** p["n"]) - p["kr"] * s[2],
        ]),
        "time_scale_hours": 100.0,
        "init": [0.3, 0.3, 0.3],
    },
    "lenbury_pacheenburawana_1991": {
        "vars": ["x", "y"],
        "params": {"a": 1.0, "b": 0.5, "c": 2.0, "d": 0.3},
        "rhs": lambda s, p, t: np.array([
            p["a"] * s[0] * (1 - s[0]) - p["c"] * s[0] * s[1],
            -p["b"] * s[1] + p["d"] * s[0] * s[1],
        ]),
        "time_scale_hours": 24.0,
        "init": [0.4, 0.2],
    },
}


def integrate(model_key: str, t_hours: np.ndarray) -> np.ndarray:
    """RK4 integration over hours. Returns (len(t), n_vars) latent trajectory."""
    cfg = MODELS[model_key]
    rhs: Callable = cfg["rhs"]
    s = np.asarray(cfg["init"], dtype=np.float64)
    out = np.zeros((len(t_hours), len(s)))
    out[0] = s
    for i in range(1, len(t_hours)):
        dt = (t_hours[i] - t_hours[i - 1]) / cfg["time_scale_hours"] / 24.0
        k1 = rhs(s, cfg["params"], 0.0)
        k2 = rhs(s + 0.5 * dt * k1, cfg["params"], 0.0)
        k3 = rhs(s + 0.5 * dt * k2, cfg["params"], 0.0)
        k4 = rhs(s + dt * k3, cfg["params"], 0.0)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s
    return out


def _missing_mask(n: int, rng: np.random.Generator, rate: float) -> np.ndarray:
    keep = rng.random(n) >= rate
    keep[0] = True
    return keep


def load_physiome(
    model_key: str,
    root: Path | None = None,
    task_id: str | None = None,
    n_entities: int = 60,
    history_hours: float = 24.0,
    horizon_hours: float = 6.0,
    target_var: str = "x",
    density: float = 4.0,   # observations per hour (nominal)
    seed: int = 0,
) -> TaskBundle:
    if model_key not in MODELS:
        raise KeyError(f"unknown Physiome model {model_key}; have {sorted(MODELS)}")
    cfg = MODELS[model_key]
    rng = np.random.default_rng(seed)
    all_vars = cfg["vars"]

    hist_t = np.linspace(0.0, history_hours, int(history_hours * density) + 1)
    fut_t = np.linspace(history_hours, history_hours + horizon_hours,
                        int(horizon_hours * density) + 1)

    ev_ent, ev_t, ev_av, ev_v, ev_val = [], [], [], [], []
    tg_ent, tg_t, tg_av, tg_v, tg_val = [], [], [], [], []
    anchors: dict[int, float] = {}

    for eid in range(n_entities):
        # entity-specific initial condition -> distinct latent trajectory
        cfg_local = dict(cfg)
        cfg_local["init"] = [c * (0.7 + 0.6 * rng.random()) for c in cfg["init"]]
        traj_h = _integrate_with(cfg_local, hist_t)
        traj_f = _integrate_from(cfg_local, hist_t[-1], fut_t)
        anchors[eid] = history_hours
        for vi, _vname in enumerate(all_vars):
            keep = _missing_mask(len(hist_t), rng, rate=0.55)
            for ti in np.where(keep)[0]:
                jitter = rng.normal(0, 0.01)
                ev_ent.append(eid); ev_t.append(hist_t[ti]); ev_av.append(hist_t[ti])
                ev_v.append(vi); ev_val.append(traj_h[ti, vi] + jitter)
        # target: only the designated target variable is scored in the future
        tv = all_vars.index(target_var)
        for ti in range(1, len(fut_t)):
            tg_ent.append(eid); tg_t.append(fut_t[ti]); tg_av.append(fut_t[ti])
            tg_v.append(tv); tg_val.append(traj_f[ti, tv])

    history = EventTable(
        entity_id=np.asarray(ev_ent, dtype=np.int64),
        event_time=np.asarray(ev_t, dtype=np.float64),
        available_time=np.asarray(ev_av, dtype=np.float64),
        variable_id=np.asarray(ev_v, dtype=np.int64),
        value=np.asarray(ev_val, dtype=np.float64),
    )
    targets = EventTable(
        entity_id=np.asarray(tg_ent, dtype=np.int64),
        event_time=np.asarray(tg_t, dtype=np.float64),
        available_time=np.asarray(tg_av, dtype=np.float64),
        variable_id=np.asarray(tg_v, dtype=np.int64),
        value=np.asarray(tg_val, dtype=np.float64),
    )

    from .p12 import split_entities
    split_ids = split_entities(sorted(anchors))
    spec = TaskSpec(
        task_id=task_id or f"physiome_{model_key}_robust",
        dataset_version=f"physiome_ode_{model_key}",
        track=Track.ROBUST_IMPROVE,
        backend="physiome",
        split_ids=split_ids,
        target_variables=[target_var],
        covariate_variables=[v for v in all_vars if v != target_var],
        cutoff_rule=f"anchor + {history_hours}h",
        history_duration=history_hours,
        horizon=horizon_hours,
        query_policy=QueryPolicy.POINT,
        query_grid=[round(float(x - history_hours), 3) for x in fut_t[1:]],
        f0_model="DLinear",
        notes=f"Mechanistic ODE {model_key}; future latent trajectory is evaluator-only.",
    )
    spec.validate()
    card = DatasetCard(
        dataset_version=spec.dataset_version,
        backend="physiome",
        variables={v: {"unit": "normalized", "kind": "instantaneous",
                       "missing_meaning": "dropped observation (synthetic)"}
                   for v in all_vars},
        time_semantics={"origin": "t=0 simulation start", "unit": "hours",
                        "time_scale_hours": cfg["time_scale_hours"],
                        "irregularity": "synthetic (random per-variable dropout)"},
        provenance=[{"name": "Physiome-ODE", "url": "https://github.com/kloetergensc/Physiome-ODE",
                     "note": "local RK4 reimplementation of the named ODE; not the Zenodo archive"}],
        unknown=["Zenodo pretrained arrays not downloaded"],
    )
    return TaskBundle(
        spec=spec, card=card, history=history, targets=targets,
        variable_names=list(all_vars), entity_anchors=anchors, train_stats={},
    )


def _integrate_with(cfg: dict, t: np.ndarray) -> np.ndarray:
    rhs = cfg["rhs"]
    s = np.asarray(cfg["init"], dtype=np.float64)
    out = np.zeros((len(t), len(s)))
    out[0] = s
    for i in range(1, len(t)):
        dt = (t[i] - t[i - 1]) / cfg["time_scale_hours"] / 24.0
        k1 = rhs(s, cfg["params"], 0.0)
        k2 = rhs(s + 0.5 * dt * k1, cfg["params"], 0.0)
        k3 = rhs(s + 0.5 * dt * k2, cfg["params"], 0.0)
        k4 = rhs(s + dt * k3, cfg["params"], 0.0)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s
    return out


def _integrate_from(cfg: dict, t0: float, t: np.ndarray) -> np.ndarray:
    """Continue integration from the state reached at end of history."""
    hist = _integrate_with(cfg, np.linspace(0.0, t0, 200))
    rhs = cfg["rhs"]
    s = hist[-1].copy()
    out = np.zeros((len(t), len(s)))
    out[0] = s
    for i in range(1, len(t)):
        dt = (t[i] - t[i - 1]) / cfg["time_scale_hours"] / 24.0
        k1 = rhs(s, cfg["params"], 0.0)
        k2 = rhs(s + 0.5 * dt * k1, cfg["params"], 0.0)
        k3 = rhs(s + 0.5 * dt * k2, cfg["params"], 0.0)
        k4 = rhs(s + dt * k3, cfg["params"], 0.0)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s
    return out
