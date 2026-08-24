"""Probe trainer (spec section 12). Scaffold stub."""
from __future__ import annotations

from typing import Any


class Trainer:
    """Trains only the shared head on frozen backbone features."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def fit(self, model, train_loader, val_loader=None, epochs: int = 60) -> dict:
        raise NotImplementedError("probe training lands with the training commit")
