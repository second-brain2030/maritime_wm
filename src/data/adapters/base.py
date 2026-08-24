"""Dataset adapter interface (spec section 4)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..manifest import TrackletManifest


class DatasetAdapter(ABC):
    """Maps raw dataset folders to normalized TrackletManifest lists."""

    dataset_name: str = "custom"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def build_manifests(self) -> list[TrackletManifest]:
        """Validate the raw layout and produce normalized manifests."""
        raise NotImplementedError
