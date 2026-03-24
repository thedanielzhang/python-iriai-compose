from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from iriai_compose.workflow import Feature


class ArtifactStore(ABC):
    """Persists workflow outputs and reusable documents."""

    @abstractmethod
    async def get(self, key: str, *, feature: Feature) -> Any | None: ...

    @abstractmethod
    async def put(self, key: str, value: Any, *, feature: Feature) -> None: ...

    @abstractmethod
    async def delete(self, key: str, *, feature: Feature) -> None: ...


class AgentSession(BaseModel):
    """Persists agent session data for continuity across invocations."""

    session_key: str
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStore(ABC):
    """Persists agent session data for continuity across invocations."""

    @abstractmethod
    async def load(self, session_key: str) -> AgentSession | None: ...

    @abstractmethod
    async def save(self, session: AgentSession) -> None: ...


class ContextProvider(ABC):
    """Resolves context keys to a prompt-ready string for agent invocations."""

    @abstractmethod
    async def resolve(self, keys: list[str], *, feature: Feature) -> str: ...


class InMemoryArtifactStore(ArtifactStore):
    """In-memory artifact store for development and testing."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def get(self, key: str, *, feature: Feature) -> Any | None:
        return self._store.get(feature.id, {}).get(key)

    async def put(self, key: str, value: Any, *, feature: Feature) -> None:
        self._store.setdefault(feature.id, {})[key] = value

    async def delete(self, key: str, *, feature: Feature) -> None:
        self._store.get(feature.id, {}).pop(key, None)


class InMemorySessionStore(SessionStore):
    """In-memory session store for development and testing."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    async def load(self, session_key: str) -> AgentSession | None:
        return self._sessions.get(session_key)

    async def save(self, session: AgentSession) -> None:
        self._sessions[session.session_key] = session


class DefaultContextProvider(ContextProvider):
    """Context provider backed by an ArtifactStore and optional static files."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        static_files: dict[str, Path] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.static_files = static_files or {}

    async def resolve(self, keys: list[str], *, feature: Feature) -> str:
        sections = []
        for key in keys:
            if key in self.static_files:
                content = self.static_files[key].read_text()
            else:
                content = await self.artifacts.get(key, feature=feature)
            if content:
                sections.append(f"## {key}\n\n{content}")
        return "\n\n---\n\n".join(sections)
