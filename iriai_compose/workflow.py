from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from iriai_compose.runner import WorkflowRunner


class Workspace(BaseModel):
    """A physical environment where agents execute."""

    id: str
    path: Path
    branch: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Feature(BaseModel):
    """A concrete execution instance binding identity to a workflow and workspace."""

    id: str
    name: str
    slug: str
    workflow_name: str
    workspace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Phase(ABC):
    """Orchestration unit. Groups tasks with control flow."""

    name: str

    async def on_start(
        self, runner: WorkflowRunner, feature: Feature, state: BaseModel
    ) -> None:
        """Called before execute. Override for setup, logging, validation."""

    async def on_done(
        self, runner: WorkflowRunner, feature: Feature, state: BaseModel
    ) -> None:
        """Called after execute with the resulting state. Override for teardown."""

    @abstractmethod
    async def execute(
        self, runner: WorkflowRunner, feature: Feature, state: BaseModel
    ) -> BaseModel: ...


class Workflow(ABC):
    """A reusable template. Sequence of Phase types."""

    name: str

    async def on_start(
        self, runner: WorkflowRunner, feature: Feature, state: BaseModel
    ) -> None:
        """Called before first phase. Override for workflow-level setup."""

    async def on_done(
        self, runner: WorkflowRunner, feature: Feature, state: BaseModel
    ) -> None:
        """Called after last phase with final state. Override for teardown."""

    @abstractmethod
    def build_phases(self) -> list[type[Phase]]: ...
