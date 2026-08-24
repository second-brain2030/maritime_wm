"""String-keyed factory registry (spec section 11: how to add a new adapter)."""
from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any] | None = None):
        """Register a factory; usable directly or as a decorator."""
        if factory is not None:
            self._factories[name] = factory
            return factory

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._factories[name] = fn
            return fn

        return deco

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self._factories:
            raise KeyError(
                f"{self.name}: unknown entry {name!r}; known: {self.names()}"
            )
        return self._factories[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories
