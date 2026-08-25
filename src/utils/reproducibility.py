"""Reproducibility helpers (spec sections 15, 16).

Every run output directory contains resolved_config.yaml, git_state.json,
environment.txt, dataset manifest hashes, and seeds. Feature caches use
content-addressed keys that include checkpoint, preprocessing, frame sampling,
and manifest hash (spec section 13).
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from importlib import metadata
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_state(repo_path: str = ".") -> dict[str, Any]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", repo_path, *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def environment_snapshot() -> dict[str, str]:
    return {
        dist.metadata["Name"]: dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }


def content_addressed_key(parts: dict[str, Any]) -> str:
    """sha256 over canonical JSON of parts; order-insensitive."""
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
