"""Model registry + limited hyper-parameter options.

The agent may only `SELECT_MODEL` from the ids here and `SET_MODEL_OPTION` from
the small `options` dict per model. No arbitrary hyper-parameter dicts.
"""
from __future__ import annotations

from typing import Any, Callable

from ..logging_utils import get_logger
from .base import Forecaster
from .heavy import HEAVY_MODELS, build as build_heavy, status as heavy_status

log = get_logger("models.registry")

# id -> (constructor, declared input interface, limited options)
INTERFACE = {
    "input_type": str,          # "event_mask" | "regular_grid"
    "accepts_mask": bool,
    "accepts_dt": bool,
}


def _catalog() -> dict[str, dict[str, Any]]:
    from .runners import LastValue, MeanValue, DLinear
    from .deep import PatchTST, ITransformer

    cat: dict[str, dict[str, Any]] = {
        "LastValue": {
            "factory": lambda cfg: LastValue(),
            "input_type": "event_mask", "accepts_mask": True, "accepts_dt": True,
            "options": {},
            "cost": "none",
            "role": "sanity baseline (excluded from the 10-model list)",
        },
        "MeanValue": {
            "factory": lambda cfg: MeanValue(),
            "input_type": "event_mask", "accepts_mask": True, "accepts_dt": False,
            "options": {},
            "cost": "none",
            "role": "sanity baseline",
        },
        "DLinear": {
            "factory": lambda cfg: DLinear(
                epochs=cfg.get("epochs", 200), lr=cfg.get("lr", 1e-2),
                kernel=cfg.get("kernel", 5), seed=cfg.get("seed", 0)),
            "input_type": "regular_grid", "accepts_mask": False, "accepts_dt": False,
            "options": {"epochs": [100, 200], "kernel": [3, 5, 7]},
            "cost": "low",
            "role": "engineering stand-in for f0 until APN is vendored",
        },
        "PatchTST": {
            "factory": lambda cfg: PatchTST(
                patch=cfg.get("patch", 4), stride=cfg.get("stride", 2),
                d_model=cfg.get("d_model", 32), n_layers=cfg.get("n_layers", 2),
                epochs=cfg.get("epochs", 120), lr=cfg.get("lr", 1e-3)),
            "input_type": "regular_grid", "accepts_mask": False, "accepts_dt": False,
            "options": {"patch": [2, 4, 8], "d_model": [16, 32, 64], "n_layers": [1, 2]},
            "cost": "medium",
            "role": "PatchTST-style re-implementation",
        },
        "iTransformer": {
            "factory": lambda cfg: ITransformer(
                d_model=cfg.get("d_model", 32), n_layers=cfg.get("n_layers", 2),
                epochs=cfg.get("epochs", 120), lr=cfg.get("lr", 1e-3)),
            "input_type": "regular_grid", "accepts_mask": False, "accepts_dt": False,
            "options": {"d_model": [16, 32, 64], "n_layers": [1, 2]},
            "cost": "medium",
            "role": "iTransformer-style re-implementation",
        },
    }
    for mid, meta in HEAVY_MODELS.items():
        cat[mid] = {
            "factory": None,
            "input_type": "event_mask", "accepts_mask": meta["needs_mask_dt"],
            "accepts_dt": meta["needs_mask_dt"],
            "options": {},
            "cost": "high",
            "role": f"{meta['paper']} {meta['kind']} model",
            "paper_url": meta["paper_url"],
            "available": False,
            "blocked_reason": "author code not vendored in this environment",
        }
    return cat


CATALOG: dict[str, dict[str, Any]] = _catalog()


def registered_ids() -> list[str]:
    return sorted(CATALOG)


def runnable_ids() -> list[str]:
    return sorted(m for m, v in CATALOG.items() if v["factory"] is not None)


def describe(model_id: str) -> dict[str, Any]:
    meta = dict(CATALOG[model_id])
    meta.pop("factory", None)
    return meta


def make(model_id: str, model_config: dict[str, Any], resources_cfg: dict[str, Any] | None = None) -> Forecaster:
    meta = CATALOG.get(model_id)
    if meta is None:
        raise ValueError(f"unknown model {model_id!r}; registered: {registered_ids()}")
    if meta["factory"] is None:
        return build_heavy(model_id, resources_cfg)  # raises NotImplementedError with reason
    for key in model_config:
        if key == "seed":          # universal: set by the harness, not an agent choice
            continue
        if key not in meta["options"]:
            raise ValueError(f"{model_id}: option {key!r} not allowed; have {sorted(meta['options'])}")
        if meta["options"][key] and model_config[key] not in meta["options"][key]:
            raise ValueError(
                f"{model_id}.{key}={model_config[key]} not in allowed {meta['options'][key]}")
    return meta["factory"](model_config)


def availability_report(resources_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"runnable": {}, "blocked": {}}
    for mid, meta in CATALOG.items():
        if meta["factory"] is not None:
            out["runnable"][mid] = describe(mid)
        else:
            st = heavy_status(mid, resources_cfg)
            out["blocked"][mid] = {**st, "paper_url": meta.get("paper_url")}
    return out
