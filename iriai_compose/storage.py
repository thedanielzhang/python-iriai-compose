from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from iriai_compose.workflow import Feature


# ---------------------------------------------------------------------------
# Store (generic persistence contract)
# ---------------------------------------------------------------------------


class Store(ABC):
    """Generic key-value persistence, scoped by feature.

    Compose defines this contract.  Applications provide backends
    (in-memory, Postgres, filesystem, etc.).  The runner holds named
    stores via ``stores: dict[str, Store]``.
    """

    @abstractmethod
    async def get(self, key: str, *, feature: Feature) -> Any | None: ...

    @abstractmethod
    async def put(self, key: str, value: Any, *, feature: Feature) -> None: ...

    @abstractmethod
    async def delete(self, key: str, *, feature: Feature) -> None: ...


class ArtifactStore(Store):
    """.. deprecated:: Use :class:`Store` instead."""

    pass


# ---------------------------------------------------------------------------
# Session persistence (deprecated — runtime concern)
# ---------------------------------------------------------------------------


class AgentSession(BaseModel):
    """Persists agent session data for continuity across invocations.

    .. deprecated:: Session persistence is a runtime concern.
       Pass session stores to your runtime directly.
    """

    session_key: str
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStore(ABC):
    """Persists agent session data for continuity across invocations.

    .. deprecated:: Session persistence is a runtime concern.
       Pass session stores to your runtime directly.
    """

    @abstractmethod
    async def load(self, session_key: str) -> AgentSession | None: ...

    @abstractmethod
    async def save(self, session: AgentSession) -> None: ...


# ---------------------------------------------------------------------------
# ContextProvider (orchestration — the runner uses this)
# ---------------------------------------------------------------------------


class ContextProvider(ABC):
    """Resolves context keys to a prompt-ready string for agent invocations."""

    @abstractmethod
    async def resolve(self, keys: list[str], *, feature: Feature) -> str: ...


# ---------------------------------------------------------------------------
# In-memory implementations
# ---------------------------------------------------------------------------


class InMemoryStore(Store):
    """In-memory store for development and testing."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def get(self, key: str, *, feature: Feature) -> Any | None:
        return self._store.get(feature.id, {}).get(key)

    async def put(self, key: str, value: Any, *, feature: Feature) -> None:
        self._store.setdefault(feature.id, {})[key] = value

    async def delete(self, key: str, *, feature: Feature) -> None:
        self._store.get(feature.id, {}).pop(key, None)


InMemoryArtifactStore = InMemoryStore
""".. deprecated:: Use :class:`InMemoryStore` instead."""


class InMemorySessionStore(SessionStore):
    """In-memory session store for development and testing.

    .. deprecated:: Session persistence is a runtime concern.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    async def load(self, session_key: str) -> AgentSession | None:
        return self._sessions.get(session_key)

    async def save(self, session: AgentSession) -> None:
        self._sessions[session.session_key] = session


# ---------------------------------------------------------------------------
# Default ContextProvider
# ---------------------------------------------------------------------------


class DefaultContextProvider(ContextProvider):
    """Context provider backed by named stores and optional static files.

    Keys can be **namespaced** (``"artifacts.prd"``) to target a specific
    store, or **plain** (``"prd"``) to scan all stores in insertion order.

    Resolution order for each key:

    1. Static files — exact match on the full key
    2. Namespaced lookup — if the key contains ``.`` and the prefix
       matches a registered store name, look up the suffix in that store
    3. Fallback scan — try every store; first non-None wins
    """

    def __init__(
        self,
        stores: dict[str, Store] | None = None,
        static_files: dict[str, Path] | None = None,
        # Deprecated
        artifacts: Store | None = None,
    ) -> None:
        if artifacts is not None:
            warnings.warn(
                "DefaultContextProvider(artifacts=) is deprecated; "
                "use stores= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            stores = stores or {}
            stores.setdefault("artifacts", artifacts)
        self.stores = stores or {}
        self.static_files = static_files or {}

    async def resolve(self, keys: list[str], *, feature: Feature) -> str:
        sections = []
        for key in keys:
            content = None
            if key in self.static_files:
                content = self.static_files[key].read_text()
            else:
                content = await self._resolve_from_stores(key, feature=feature)
            if content:
                sections.append(f"## {key}\n\n{content}")
        return "\n\n---\n\n".join(sections)

    async def _resolve_from_stores(
        self, key: str, *, feature: Feature
    ) -> Any | None:
        # Namespaced: "artifacts.prd" → look up "prd" in "artifacts" store
        dot = key.find(".")
        if dot > 0:
            store_name, store_key = key[:dot], key[dot + 1:]
            if store_name in self.stores:
                return await self.stores[store_name].get(
                    store_key, feature=feature
                )

        # Plain key or prefix didn't match — scan all stores
        for store in self.stores.values():
            content = await store.get(key, feature=feature)
            if content is not None:
                return content
        return None
