"""YAML config composition (spec section 11: Hydra or equivalent).

Experiment configs may ``extends: <parent>`` for deep-merged defaults; the
fully resolved config is dumped to every output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(category: str, name: str, _seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Load configs/<category>/<name>.yaml, resolving ``extends`` chains."""
    path = CONFIG_ROOT / category / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    extends = raw.pop("extends", None)
    if extends:
        if extends in _seen:
            raise ValueError(f"config extends cycle involving {extends!r}")
        parent = load_config(category, extends, _seen | {name})
        return deep_merge(parent, raw)
    return raw


def resolve_config_path(category: str, name: str) -> Path:
    return CONFIG_ROOT / category / f"{name}.yaml"


def dump_resolved_config(cfg: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
