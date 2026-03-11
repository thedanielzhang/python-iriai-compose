from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from iriai_compose.actors import AgentActor, InteractionActor
from iriai_compose.exceptions import IriaiError, ResolutionError, TaskExecutionError
from iriai_compose.pending import Pending
from iriai_compose.storage import ArtifactStore, ContextProvider, SessionStore

if TYPE_CHECKING:
    from iriai_compose.actors import Actor, Role
    from iriai_compose.tasks import Task
    from iriai_compose.workflow import Feature, Workflow, Workspace


def _extract_agent_actors(task: Task) -> list[AgentActor]:
    """Extract all AgentActor references from a task for collision detection."""
    actors = []
    for field in ("actor", "questioner", "responder", "approver", "chooser"):
        val = getattr(task, field, None)
        if isinstance(val, AgentActor):
            actors.append(val)
    return actors

# Context variable for phase name — safe under concurrent async workflows.
_current_phase_var: ContextVar[str] = ContextVar("_current_phase", default="")


class AgentRuntime(ABC):
    """Executes agent invocations."""

    name: str

    @abstractmethod
    async def invoke(
        self,
        role: Role,
        prompt: str,
        *,
        output_type: type[BaseModel] | None = None,
        workspace: Workspace | None = None,
        session_key: str | None = None,
    ) -> str | BaseModel: ...


class InteractionRuntime(ABC):
    """Resolves interaction requests."""

    name: str

    @abstractmethod
    async def resolve(self, pending: Pending) -> str | bool: ...


class WorkflowRunner(ABC):
    """The coordinator. Phases call runner.run(task). Tasks call runner.resolve(actor)."""

    artifacts: ArtifactStore
    sessions: SessionStore | None
    context_provider: ContextProvider
    services: dict[str, Any]

    async def run(self, task: Task, feature: Feature, *, phase_name: str = "") -> Any:
        """Execute a task. The task defines the interaction pattern."""
        if phase_name:
            _current_phase_var.set(phase_name)
        try:
            return await task.execute(self, feature)
        except IriaiError:
            raise
        except Exception as e:
            raise TaskExecutionError(
                task=task,
                feature=feature,
                phase_name=_current_phase_var.get(),
            ) from e

    @abstractmethod
    async def resolve(
        self,
        actor: Actor,
        prompt: str,
        *,
        feature: Feature,
        context_keys: list[str] | None = None,
        output_type: type[BaseModel] | None = None,
        kind: Literal["approve", "choose", "respond"] | None = None,
        options: list[str] | None = None,
    ) -> Any: ...

    async def parallel(
        self,
        tasks: list[Task],
        feature: Feature,
        *,
        fail_fast: bool = True,
    ) -> list[Any]:
        """Run tasks concurrently.

        fail_fast=True (default): first exception cancels remaining tasks.
        fail_fast=False: all tasks run to completion; exceptions are collected
        and raised as an ExceptionGroup.
        """
        # Enforce: same AgentActor must not run concurrently (session collision)
        seen_agents: set[str] = set()
        for task in tasks:
            for actor in _extract_agent_actors(task):
                if actor.name in seen_agents:
                    raise ValueError(
                        f"Actor '{actor.name}' used in multiple parallel tasks. "
                        f"Define separate actors for parallel work."
                    )
                seen_agents.add(actor.name)

        if fail_fast:
            results: list[Any] = [None] * len(tasks)
            async with asyncio.TaskGroup() as tg:
                async def _run(idx: int, task: Task) -> None:
                    results[idx] = await task.execute(self, feature)

                for i, task in enumerate(tasks):
                    tg.create_task(_run(i, task))
            return results
        else:
            gathered = await asyncio.gather(
                *[task.execute(self, feature) for task in tasks],
                return_exceptions=True,
            )
            exceptions = [r for r in gathered if isinstance(r, BaseException)]
            if exceptions:
                raise ExceptionGroup("parallel task failures", exceptions)
            return list(gathered)

    async def execute_child(
        self,
        workflow: Workflow,
        feature: Feature,
        state: BaseModel,
        *,
        workspace_id: str | None = None,
    ) -> BaseModel:
        """Execute a child workflow, optionally rebinding workspace."""
        if workspace_id is not None:
            feature = feature.model_copy(update={"workspace_id": workspace_id})
        return await self.execute_workflow(workflow, feature, state)

    async def execute_workflow(
        self,
        workflow: Workflow,
        feature: Feature,
        state: BaseModel,
    ) -> BaseModel:
        """Execute a workflow's phases in sequence."""
        for phase_cls in workflow.build_phases():
            phase = phase_cls()
            _current_phase_var.set(phase.name)
            state = await phase.execute(self, feature, state)
        return state

    def get_workspace(self, workspace_id: str | None) -> Workspace | None:
        """Look up a workspace by ID."""
        return None


class DefaultWorkflowRunner(WorkflowRunner):
    """Default implementation of WorkflowRunner."""

    def __init__(
        self,
        *,
        agent_runtime: AgentRuntime,
        interaction_runtimes: dict[str, InteractionRuntime],
        artifacts: ArtifactStore,
        sessions: SessionStore | None = None,
        context_provider: ContextProvider,
        workspaces: dict[str, Workspace] | None = None,
        services: dict[str, Any] | None = None,
    ) -> None:
        self.agent_runtime = agent_runtime
        self.interaction_runtimes = interaction_runtimes
        self.artifacts = artifacts
        self.sessions = sessions
        self.context_provider = context_provider
        self._workspaces = workspaces or {}
        self.services = services or {}

    def _resolve_interaction_runtime(self, resolver: str) -> InteractionRuntime:
        """Route a resolver key to an InteractionRuntime.
        Tries exact match first, then prefix match."""
        if resolver in self.interaction_runtimes:
            return self.interaction_runtimes[resolver]
        prefix = resolver.split(".")[0]
        if prefix in self.interaction_runtimes:
            return self.interaction_runtimes[prefix]
        raise ResolutionError(
            f"No InteractionRuntime registered for resolver '{resolver}'"
        )

    async def resolve(
        self,
        actor: Actor,
        prompt: str,
        *,
        feature: Feature,
        context_keys: list[str] | None = None,
        output_type: type[BaseModel] | None = None,
        kind: Literal["approve", "choose", "respond"] | None = None,
        options: list[str] | None = None,
    ) -> Any:
        if isinstance(actor, AgentActor):
            # Merge context: actor baseline + task-specific, deduplicated
            all_keys = list(
                dict.fromkeys(actor.context_keys + (context_keys or []))
            )
            context_str = ""
            if all_keys:
                context_str = await self.context_provider.resolve(
                    all_keys, feature=feature
                )
            full_prompt = (
                f"{context_str}\n\n## Task\n{prompt}" if context_str else prompt
            )

            # Session key derived from actor identity + feature
            session_key = f"{actor.name}:{feature.id}"

            # Dispatch to agent runtime
            workspace = self.get_workspace(feature.workspace_id)
            return await self.agent_runtime.invoke(
                role=actor.role,
                prompt=full_prompt,
                output_type=output_type,
                workspace=workspace,
                session_key=session_key,
            )

        elif isinstance(actor, InteractionActor):
            # Create Pending and dispatch to the correct interaction runtime
            pending = Pending(
                id=str(uuid4()),
                feature_id=feature.id,
                phase_name=_current_phase_var.get(),
                kind=kind or "respond",
                prompt=prompt,
                options=options,
                created_at=datetime.now(),
            )
            runtime = self._resolve_interaction_runtime(actor.resolver)
            return await runtime.resolve(pending)

        raise ResolutionError(f"Unknown actor type: {type(actor).__name__}")

    def get_workspace(self, workspace_id: str | None) -> Workspace | None:
        if workspace_id is None:
            return None
        return self._workspaces.get(workspace_id)
