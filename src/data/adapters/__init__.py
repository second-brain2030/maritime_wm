"""Dataset adapters (spec section 4: map upstream folder names via config)."""
from __future__ import annotations

from utils.registry import Registry

adapter_registry = Registry("dataset_adapters")


def get_adapter(name: str, config: dict):
    """Return an adapter instance for dataset ``name``; fails loudly when absent."""
    if name not in adapter_registry:
        raise NotImplementedError(
            f"no adapter registered for dataset {name!r}; known: {adapter_registry.names()} "
            "(implement data/adapters to ingest the raw dataset)"
        )
    return adapter_registry.create(name, config=config)
