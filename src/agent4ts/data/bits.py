"""BITS upstream bridge.

BITS is the design's intended unified substrate for track A, but it pins
Python >=3.8,<3.11 with torch 2.4-2.5, which conflicts with the `tim` env
(Python 3.11, torch 2.13) that runs everything else. Rather than silently
substituting data, this bridge:
  * discovers the vendored BITS source and reports its version,
  * exposes the external-model adapter contract (build_model / forward_batch),
  * returns `blocked_resource` for anything that would require the uninstalled
    BITS runtime, so callers degrade explicitly instead of pretending.

The P12 loader is the live native path for track A.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger

log = get_logger("data.bits")

BITS_PY_REQUIRES = ">=3.8,<3.11"


class BlockedResource(RuntimeError):
    """Raised when a BITS operation needs the uninstalled upstream runtime."""

    def __init__(self, what: str, reason: str):
        self.what = what
        self.reason = reason
        super().__init__(f"blocked_resource: {what} ({reason})")

    def as_status(self) -> dict[str, Any]:
        return {"status": "blocked_resource", "what": self.what, "reason": self.reason}


def discover_source(bits_src: Path) -> dict[str, Any]:
    """Report what of the vendored BITS snapshot is actually present."""
    info: dict[str, Any] = {
        "path": str(bits_src),
        "present": bits_src.exists(),
        "package_version": None,
        "python_requires": BITS_PY_REQUIRES,
        "installed": False,
        "dataset_configs": [],
    }
    if not bits_src.exists():
        return info
    pyproject = bits_src / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("version") and info["package_version"] is None:
                info["package_version"] = s.split("=", 1)[-1].strip().strip('"').strip("'")
            if s.startswith("requires-python"):
                info["python_requires"] = s.split("=", 1)[-1].strip().strip('"').strip("'")
    cfgs = bits_src / "dataset_configs"
    if cfgs.exists():
        info["dataset_configs"] = sorted(p.name for p in cfgs.glob("*.yaml"))
    info["installed"] = _bits_importable()
    return info


def _bits_importable() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("bits") is not None
    except Exception:
        return False


def check_environment() -> dict[str, Any]:
    """Is the current interpreter able to run BITS at all?"""
    import sys
    ok = (3, 8) <= sys.version_info[:2] < (3, 11)
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "satisfies_requires_python": ok,
        "required": BITS_PY_REQUIRES,
        "importable": _bits_importable(),
    }


def run_cli(args: list[str], bits_src: Path, timeout: int = 600) -> dict[str, Any]:
    """Invoke `bits ...` in the vendored source dir. Blocks unless installed."""
    env = check_environment()
    if not env["satisfies_requires_python"] or not env["importable"]:
        raise BlockedResource(
            f"bits {' '.join(args)}",
            f"python {env['python']} does not satisfy {BITS_PY_REQUIRES} or package not installed",
        )
    proc = subprocess.run(
        ["bits", *args], cwd=str(bits_src), capture_output=True, text=True, timeout=timeout
    )
    return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def dataset_catalog(bits_src: Path) -> dict[str, Any]:
    """Read the vendored catalog for declared datasets/versions, no download."""
    manifest = bits_src / "dataset_configs" / "processed-v1-manifest.json"
    if not manifest.exists():
        legacy = bits_src / "dataset_configs" / "legacy" / "processed-v1-manifest.json"
        manifest = legacy if legacy.exists() else manifest
    if not manifest.exists():
        return {"status": "missing", "declared": []}
    with manifest.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {"status": "ok", "declared": sorted(data.keys()) if isinstance(data, dict) else []}


def load_bits_task(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Track-A-via-BITS entry point. Always blocked until upstream is installed."""
    raise BlockedResource(
        "load_bits_task",
        "BITS runtime not installed (Python 3.10 + torch 2.4-2.5 required); "
        "native P12 backend is the active track-A path",
    )
