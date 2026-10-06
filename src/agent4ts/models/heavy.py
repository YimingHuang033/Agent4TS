"""Availability registry for the heavy 2026 irregular-series models.

DESIGN section 3 lists ten models. Five are implemented natively in this repo
(runners.py / deep.py). The rest are published only as author code that is not
vendored here and requires either a separate environment or an install step.
This module states, per model, exactly what exists — no stand-in is registered
under a real model's name.

To actually enable one, set its `local_dir` in config/<kind>/resources.yaml to a
checked-out copy of the author repository and add a thin adapter; `status` flips
from `not_vendored` to `vendored` automatically when the directory appears.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

log = get_logger("models.heavy")

# model_id -> metadata. `paper_url` is source traceability, not a claim of local
# availability. `implies` is the interface the adapter must satisfy.
HEAVY_MODELS: dict[str, dict[str, Any]] = {
    "TFMixer": {
        "kind": "irregular",
        "paper": "ICML 2026",
        "paper_url": "https://github.com/decisionintelligence/TFMixer",
        "needs_mask_dt": True,
        "note": "external-model adapter only; not vendored",
    },
    "KAFNet": {
        "kind": "irregular",
        "paper": "AAAI 2026",
        "paper_url": "https://github.com/zhouziyu02/KAFNet",
        "needs_mask_dt": True,
        "note": "not vendored; needs author repo",
    },
    "APN": {
        "kind": "irregular",
        "paper": "AAAI 2026",
        "paper_url": "https://github.com/decisionintelligence/APN",
        "needs_mask_dt": True,
        "note": "design's default f0; not vendored, DLinear stands in until then",
    },
    "ASTGI": {
        "kind": "irregular",
        "paper": "ICLR 2026",
        "paper_url": "https://github.com/decisionintelligence/ASTGI",
        "needs_mask_dt": True,
        "note": "not vendored",
    },
    "TiWeaver": {
        "kind": "irregular_async",
        "paper": "KDD 2026",
        "paper_url": "https://arxiv.org/abs/2606.03121",
        "archive": "https://doi.org/10.5281/zenodo.20424563",
        "needs_mask_dt": True,
        "note": "code archive via Zenodo; not vendored",
    },
    "HyperIMTS": {
        "kind": "irregular",
        "paper": "ICML 2025",
        "paper_url": "https://github.com/Ladbaby/PyOmniTS",
        "needs_mask_dt": True,
        "note": "author implementation in PyOmniTS; not vendored",
    },
    "tPatchGNN": {
        "kind": "irregular",
        "paper": "ICML 2024",
        "paper_url": "https://github.com/usail-hkust/t-PatchGNN",
        "needs_mask_dt": True,
        "note": "not vendored",
    },
}


def status(model_id: str, resources_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = HEAVY_MODELS.get(model_id)
    if meta is None:
        return {"model_id": model_id, "status": "unknown"}
    local = (resources_cfg or {}).get("heavy_model_dirs", {}).get(model_id)
    if local and Path(local).exists():
        return {"model_id": model_id, "status": "vendored", "local_dir": local, **meta}
    return {
        "model_id": model_id,
        "status": "not_vendored",
        "blocked_reason": "author code not present in this environment",
        **meta,
    }


def all_statuses(resources_cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return {m: status(m, resources_cfg) for m in HEAVY_MODELS}


def build(model_id: str, resources_cfg: dict[str, Any] | None = None):
    """Return a heavy model instance, or explain precisely why it cannot exist."""
    st = status(model_id, resources_cfg)
    if st["status"] != "vendored":
        raise NotImplementedError(
            f"{model_id} is not available: {st.get('blocked_reason')}. "
            f"Source: {st.get('paper_url')}. Set resources.heavy_model_dirs.{model_id} "
            f"in config to a checked-out author repo to enable it."
        )
    raise NotImplementedError(f"{model_id} directory present but no adapter registered yet")
