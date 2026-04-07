from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(BaseModel):
    """Defines expertise, perspective, and capabilities for an AgentActor."""

    name: str
    prompt: str
    tools: list[str] = Field(default_factory=list)
    model: str | None = None
    effort: Literal["low", "medium", "high", "max"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Actor(BaseModel):
    """Any entity that can receive a prompt and produce a response."""

    name: str


class AgentActor(Actor):
    """Resolved by an AgentRuntime."""

    role: Role
    context_keys: list[str] = Field(default_factory=list)
    persistent: bool = True
    resolver: str = "agent"


class InteractionActor(Actor):
    """Resolved by an InteractionRuntime."""

    resolver: str
