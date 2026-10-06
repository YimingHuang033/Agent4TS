"""Config resolution.

Every position, key, threshold and path lives in `config/`. This module only
loads and merges those YAML files; it never hard-codes a resource path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(name: str = "default", extra: list[str] | None = None) -> dict[str, Any]:
    """Load config/<name>.yaml, merging on top of config/default.yaml.

    `extra` lists further config stems merged last, in order. Sub-folder configs
    are given as "smoke/foo" etc.
    """
    merged = _read_yaml(CONFIG_DIR / "default.yaml")
    if name != "default":
        merged = _deep_merge(merged, _read_yaml(CONFIG_DIR / f"{name}.yaml"))
    for stem in extra or []:
        merged = _deep_merge(merged, _read_yaml(CONFIG_DIR / f"{stem}.yaml"))
    _resolve_paths(merged)
    return merged


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Turn relative path entries into absolute paths under the project root."""
    paths = cfg.setdefault("paths", {})
    for key, val in list(paths.items()):
        if isinstance(val, str) and not os.path.isabs(val):
            paths[key] = str((PROJECT_ROOT / val).resolve())
    # env var wins so a different machine can relocate the data cache
    if os.environ.get("ATS_DATA_ROOT"):
        paths["data_root"] = os.environ["ATS_DATA_ROOT"]


def experiment_dir(cfg: dict[str, Any], kind: str) -> Path:
    """Return (and create) the results/log sub-dir for an experiment kind."""
    base = Path(cfg["paths"]["results_root"]) / kind
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_dir(cfg: dict[str, Any], kind: str) -> Path:
    base = Path(cfg["paths"]["log_root"]) / kind
    base.mkdir(parents=True, exist_ok=True)
    return base


def llm_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    llm = dict(cfg.get("llm", {}))
    env_key = llm.pop("api_key_env", None)
    if env_key:
        llm["api_key"] = os.environ.get(env_key, "")
    else:
        llm["api_key"] = llm.get("api_key", "")
    base_url = os.environ.get(llm.get("base_url_env", "") or "", "") or llm.get("base_url", "")
    llm["base_url"] = base_url
    return llm
