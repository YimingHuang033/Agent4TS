"""Data backends. One entry point: `load_task`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from ..schemas import TaskBundle

log = get_logger("data")


def load_task(spec: dict[str, Any], cfg: dict[str, Any]) -> TaskBundle:
    """Dispatch to a backend by `spec.backend`. No silent substitution."""
    backend = spec["backend"]
    paths = cfg["paths"]
    if backend == "p12":
        from .p12 import load_p12
        return load_p12(
            root=Path(spec.get("root", paths["p12_root"])),
            task_id=spec.get("task_id", "p12_native_irr"),
            track=_track(spec),
            history_hours=spec.get("history_hours", 24.0),
            horizon_hours=spec.get("horizon_hours", 6.0),
            target_variables=tuple(spec.get("target_variables", ("HR", "SysABP", "Temp", "Urine"))),
            max_entities=spec.get("max_entities"),
        )
    if backend == "physiome":
        from .physiome import load_physiome
        return load_physiome(
            model_key=spec["model_key"],
            root=Path(paths["physiome_root"]),
            task_id=spec.get("task_id"),
            n_entities=spec.get("n_entities", 60),
            history_hours=spec.get("history_hours", 24.0),
            horizon_hours=spec.get("horizon_hours", 6.0),
            target_var=spec.get("target_var", "x"),
            seed=spec.get("seed", 0),
        )
    if backend == "timeimm":
        from .timeimm import load_epa_air
        return load_epa_air(
            root=Path(spec.get("root", paths["timeimm_root"])),
            city=spec.get("city", "Los_Angeles"),
            track=_track(spec),
            history_days=spec.get("history_days", 7.0),
            horizon_days=spec.get("horizon_days", 1.0),
            target_variables=tuple(spec.get("target_variables", ("O3", "PM2.5"))),
            max_entities=spec.get("max_entities", 400),
        )
    if backend == "ushcn":
        from .ushcn import load_ushcn
        return load_ushcn(spec, cfg)
    if backend == "pamap2":
        from .pamap2 import load_pamap2
        return load_pamap2(spec, cfg)
    if backend == "bits":
        from .bits import load_bits_task
        return load_bits_task(spec, cfg)  # raises BlockedResource
    if backend == "mimic":
        from .mimic import load_mimic
        return load_mimic(spec, cfg)  # raises BlockedAccess
    raise ValueError(f"unknown backend {backend!r}")


def _track(spec: dict[str, Any]):
    from ..schemas import Track
    return Track(spec.get("track", "native_irr"))
