"""Fault injection / detection placeholder.

DESIGN 1 reserves three fault classes and their detection interface but disables
them for this release: no fault tasks are generated and no detection rate is
reported. This module keeps the interface so the capability can be filled in
later, and every entry point raises `NotImplementedError` with the exact reason.
"""
from __future__ import annotations

from typing import Any

FAULT_CLASSES = {
    "time_unit_misread": "history time interpreted in the wrong unit / span",
    "missing_placeholder_misuse": "a missing-value sentinel (-1, -9999) read as real data",
    "future_leakage": "information from after the cutoff used in history",
}

STATUS = "disabled"


def inject_fault(*_args: Any, **_kwargs: Any):
    raise NotImplementedError(
        f"fault injection is {STATUS} in this release (DESIGN 1: placeholder only)"
    )


def detect_irr_fault(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {"status": "not_implemented", "enabled": False, "classes": FAULT_CLASSES}


def is_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("fault_detection", False))


def describe() -> dict[str, Any]:
    return {"status": STATUS, "classes": FAULT_CLASSES,
            "note": "no fault tasks are generated; no detection rate is reported"}
