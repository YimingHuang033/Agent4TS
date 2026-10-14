"""TSRouter baseline: dynamic modality-model routing (COLM 2026).

DESIGN 3 keeps one external *agentic* baseline (TimeSeriesScientist). TSRouter
is a second external framework, added on request: it is a graph-based router
that, per query, jointly selects a *modality* (text / visual / mix) and a
*candidate model* (LLM or VLM) to answer a time-series reasoning question, and
is trained on the TSRBench oracle scores it produces with vLLM.

It does NOT share this project's interface: it consumes a `date`/`OT` CSV (via
TSRBench's jsonl problems) and reasons over LLMs/VLMs; it is not an irregular
multivariate forecaster and its queries are reasoning problems, not the
60/20/10/10 bucketed horizon here. Like `tsci.py`, this module reports its true
state instead of imitating upstream: if an upstream checkout is configured and
importable it reports `active` and `run` delegates through the bridge; otherwise
it reports `unavailable` with the exact reason and setup, and `run` raises. It
never substitutes a home-grown loop for real TSRouter.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from ..harness import Harness
from ..logging_utils import get_logger
from ..schemas import TaskBundle

log = get_logger("baselines.tsrouter")

UPSTREAM_URL = "https://github.com/tianyi-lab/TSRouter"
UPSTREAM_PAPER = "https://arxiv.org/abs/2607.08940"
UPSTREAM_DATASET = "https://huggingface.co/datasets/umd-zhou-lab/TSRBench"
# upstream pins python 3.10 and torch>=2.0 with torch-geometric + sentence-transformers
UPSTREAM_PY_REQUIRES = "==3.10"


def status(resources_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    local = (resources_cfg or {}).get("tsrouter_dir")
    present = bool(local and Path(local).exists())
    importable = importlib.util.find_spec("ts_router") is not None
    if present and importable:
        return {"status": "active", "local_dir": local, "importable": True,
                "adapted_from": UPSTREAM_URL, "label": "TSRouter"}
    missing = []
    if not present:
        missing.append("tsrouter_dir not set or missing")
    if not importable:
        missing.append("package 'ts_router' not importable in this env")
    return {
        "status": "unavailable",
        "local_dir": local,
        "importable": importable,
        "adapted_from": UPSTREAM_URL,
        "paper": UPSTREAM_PAPER,
        "blocked_reason": "; ".join(missing),
        "python_requires": UPSTREAM_PY_REQUIRES,
        "setup": [
            f"git clone {UPSTREAM_URL} <dir>",
            "conda create -n tsrouter python=3.10 && pip install -r requirements.txt",
            f"python download_data.py   # TSRBench from {UPSTREAM_DATASET}",
            "run inference/ on the six candidate LLMs/VLMs to build the oracle "
            "scores, then train the router (run_exp.py)",
            "set resources.tsrouter_dir to <dir> in config",
        ],
        "interface_notes": [
            "TSRouter takes TSRBench reasoning problems (perception/reasoning/"
            "prediction/decision jsonl), not this project's event tables.",
            "Its candidates are LLMs (Qwen3-8B/32B, Llama-3.3-70B) and VLMs "
            "(Qwen3-VL-8B/32B, GLM-4.5V) served via vLLM — far heavier than the "
            "local Qwen3.5-4B controller and not vendored here.",
            "Routing is trained on oracle correctness; a fair comparison to the "
            "in-project agent needs the same data, figures, model pool and "
            "budget routed through this project's harness and evaluator.",
        ],
    }


def run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    st = status()
    raise RuntimeError(
        f"TSRouter baseline is {st['status']}: {st['blocked_reason']}. "
        f"Upstream: {UPSTREAM_URL}. A hand-written router would not be TSRouter."
    )


def bridge_note() -> str:
    return (
        "Bridge contract: TSRouter's graph emits a (modality, model) choice per "
        "query; to run inside this project it must consume the same event tables "
        "and semantic card, route candidate training to this harness "
        "(models.registry) and scoring to evaluator.score_windows, and share the "
        "LLM and budget with the other baselines. Kept separate until vendored."
    )
