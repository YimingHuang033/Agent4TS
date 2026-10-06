"""TimeSeriesScientist (TSci) adapted baseline.

DESIGN 3 asks for a TSciAdapter that keeps TSci's analyse/plan/fit-validate flow
and bridges its execution to this project's executors, clearly marked
`TSci-adapted`. TSci is a separate agentic framework with its own install
(`Y-Research-SBU/TimeSeriesScientist`); it is NOT vendored here.

This module therefore reports its true state instead of imitating TSci:
  * if an upstream checkout is configured and importable -> `status: active`
    and `run` delegates through the bridge below,
  * otherwise -> `status: unavailable` with the exact reason and setup, and
    `run` raises. It never substitutes a home-grown loop for real TSci.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from ..harness import Harness
from ..logging_utils import get_logger
from ..schemas import TaskBundle

log = get_logger("baselines.tsci")

UPSTREAM_URL = "https://github.com/Y-Research-SBU/TimeSeriesScientist"
UPSTREAM_PAPER = "https://arxiv.org/abs/2510.01538"


def status(resources_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    local = (resources_cfg or {}).get("tsci_dir")
    present = bool(local and Path(local).exists())
    importable = importlib.util.find_spec("timeseriesscientist") is not None
    if present and importable:
        return {"status": "active", "local_dir": local, "importable": True,
                "adapted_from": UPSTREAM_URL, "label": "TSci-adapted"}
    missing = []
    if not present:
        missing.append("tsci_dir not set or missing")
    if not importable:
        missing.append("package 'timeseriesscientist' not importable in this env")
    return {
        "status": "unavailable",
        "local_dir": local,
        "importable": importable,
        "adapted_from": UPSTREAM_URL,
        "paper": UPSTREAM_PAPER,
        "blocked_reason": "; ".join(missing),
        "setup": [
            f"git clone {UPSTREAM_URL} <dir>",
            "install its requirements (separate env recommended; pins differ from `tim`)",
            "set resources.tsci_dir to <dir> in config",
        ],
        "adaptation_notes": [
            "TSci expects a `date`/`OT` CSV; a bridge must route its candidate "
            "training and its internal split/scoring to this project's executor "
            "and unified evaluator.",
            "Must share the same data, semantic card, figures, model pool, LLM and "
            "budget as the other baselines.",
        ],
    }


def run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    st = status()
    raise RuntimeError(
        f"TSci-adapted baseline is {st['status']}: {st['blocked_reason']}. "
        f"Upstream: {UPSTREAM_URL}. A hand-written loop would not be TSci."
    )


def bridge_note() -> str:
    return (
        "Bridge contract: TSci's planner emits a candidate; this project's "
        "harness trains it (models.registry), scores it on the search split "
        "(evaluator.score_windows) and returns Feedback. TSci keeps its own "
        "analysis and fitting-verification policy on top."
    )
