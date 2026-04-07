from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from iriai_compose.actors import AgentActor, InteractionActor
from iriai_compose.exceptions import IriaiError, ResolutionError, TaskExecutionError
from iriai_compose.runtime import Runtime
from iriai_compose.storage import ContextProvider, DefaultContextProvider, Store

if TYPE_CHECKING:
    from iriai_compose.actors import Actor, Role
    from iriai_compose.tasks import Ask, Task
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


# ---------------------------------------------------------------------------
# Runtime subclasses
# ---------------------------------------------------------------------------


class AgentRuntime(Runtime):
    """Runtime for agent (LLM) invocations.

    New implementations should override ``ask()``.  Legacy implementations
    that override ``invoke()`` continue to work — ``ask()`` bridges to
    ``invoke()`` with a deprecation warning.
    """

    async def ask(self, task: Ask, **kwargs: Any) -> Any:
        # Bridge: if a subclass still overrides the legacy invoke(), delegate.
        if type(self).invoke is not AgentRuntime.invoke:
            warnings.warn(
                "AgentRuntime.invoke() is deprecated. "
                "Implement ask() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Build prompt the way the old framework did
            context = kwargs.get("context", "")
            combined = task.to_prompt()
            prompt = (
                f"{context}\n\n## Task\n{combined}" if context else combined
            )
            return await self.invoke(
                role=task.actor.role,
                prompt=prompt,
                output_type=task.output_type,
                workspace=kwargs.get("workspace"),
                session_key=kwargs.get("session_key"),
            )
        raise NotImplementedError(
            f"{type(self).__name__} must implement ask()"
        )

    async def invoke(
        self,
        role: Role,
        prompt: str,
        *,
        output_type: type[BaseModel] | None = None,
        workspace: Workspace | None = None,
        session_key: str | None = None,
    ) -> str | BaseModel:
        """Deprecated — implement ``ask()`` instead."""
        raise NotImplementedError(
            "invoke() is deprecated. Implement ask() instead."
        )


class InteractionRuntime(Runtime):
    """Runtime for human interactions.

    Subclasses implement ``ask()`` and decide their own presentation
    strategy by inspecting the Ask task's fields (``task.prompt``,
    ``task.input``, ``task.input_type``, ``task.output_type``, etc.).
    """

    pass


# ---------------------------------------------------------------------------
# WorkflowRunner
# ---------------------------------------------------------------------------


class WorkflowRunner(ABC):
    """The coordinator. Phases call ``runner.run(task)``.  Leaf tasks call
    ``runner.resolve(task, feature)``."""

    stores: dict[str, Store]
    context_provider: ContextProvider
    services: dict[str, Any]

    async def run(
        self, task: Task, feature: Feature, *, phase_name: str = "", **kwargs: Any
    ) -> Any:
        """Execute a task with lifecycle hooks.

        Extra ``**kwargs`` are forwarded to ``task.execute()`` so that
        runtime-specific hints can flow from phases through composite
        tasks down to the runtime.
        """
        if phase_name:
            _current_phase_var.set(phase_name)
        await task.on_start(self, feature)
        try:
            result = await task.execute(self, feature, **kwargs)
        except IriaiError as e:
            await task.on_done(self, feature, error=e)
            raise
        except Exception as e:
            wrapped = TaskExecutionError(
                task=task,
                feature=feature,
                phase_name=_current_phase_var.get(),
            )
            await task.on_done(self, feature, error=wrapped)
            raise wrapped from e
        else:
            await task.on_done(self, feature, result=result)
            return result

    @abstractmethod
    async def resolve(self, task: Ask, feature: Feature, **kwargs: Any) -> Any:
        """Dispatch an Ask to the appropriate runtime.

        Only Ask tasks reach this method — composite tasks decompose
        into Asks via ``runner.run()`` in their ``execute()``.

        Extra ``**kwargs`` from ``runner.run()`` are forwarded through
        ``Ask.execute()`` and merged into the runtime call.
        """
        ...

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
                    results[idx] = await self.run(task, feature)

                for i, task in enumerate(tasks):
                    tg.create_task(_run(i, task))
            return results
        else:
            gathered = await asyncio.gather(
                *[self.run(task, feature) for task in tasks],
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
        await workflow.on_start(self, feature, state)
        try:
            for phase_cls in workflow.build_phases():
                phase = phase_cls()
                _current_phase_var.set(phase.name)
                await phase.on_start(self, feature, state)
                try:
                    state = await phase.execute(self, feature, state)
                except Exception:
                    await phase.on_done(self, feature, state)
                    raise
                await phase.on_done(self, feature, state)
        except Exception:
            await workflow.on_done(self, feature, state)
            raise
        await workflow.on_done(self, feature, state)
        return state

    def get_workspace(self, workspace_id: str | None) -> Workspace | None:
        """Look up a workspace by ID."""
        return None


# ---------------------------------------------------------------------------
# DefaultWorkflowRunner
# ---------------------------------------------------------------------------


class DefaultWorkflowRunner(WorkflowRunner):
    """Default implementation of WorkflowRunner."""

    def __init__(
        self,
        *,
        runtimes: dict[str, Runtime] | None = None,
        stores: dict[str, Store] | None = None,
        context_provider: ContextProvider | None = None,
        workspaces: dict[str, Workspace] | None = None,
        services: dict[str, Any] | None = None,
        # --- Deprecated params (backwards compat) ---
        artifacts: Store | None = None,
        sessions: Any = None,
        agent_runtime: AgentRuntime | None = None,
        interaction_runtimes: dict[str, InteractionRuntime] | None = None,
    ) -> None:
        if agent_runtime is not None or interaction_runtimes is not None:
            warnings.warn(
                "agent_runtime / interaction_runtimes are deprecated. "
                "Use runtimes={'key': runtime, ...} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            runtimes = runtimes or {}
            if agent_runtime:
                runtimes.setdefault("agent", agent_runtime)
            if interaction_runtimes:
                runtimes.update(interaction_runtimes)

        # Handle deprecated artifacts → stores
        if artifacts is not None:
            warnings.warn(
                "artifacts= is deprecated. "
                "Use stores={'artifacts': store} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if stores is None:
                stores = {}
            stores.setdefault("artifacts", artifacts)

        # Handle deprecated sessions
        if sessions is not None:
            warnings.warn(
                "sessions= is deprecated. "
                "Pass session store to your runtime directly.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._runtimes = runtimes or {}
        self.stores = stores or {}
        self.context_provider = context_provider or DefaultContextProvider(
            stores=self.stores,
        )
        self._workspaces = workspaces or {}
        self.services = services or {}
        self.sessions = sessions  # deprecated, kept for backward compat

    @property
    def artifacts(self) -> Store:
        """.. deprecated:: Use ``runner.stores['artifacts']`` instead."""
        warnings.warn(
            "runner.artifacts is deprecated. "
            "Use runner.stores['artifacts'] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "artifacts" not in self.stores:
            raise AttributeError(
                "No 'artifacts' store registered. Use runner.stores instead."
            )
        return self.stores["artifacts"]

    def _resolve_runtime(self, resolver: str) -> Runtime:
        """Route a resolver key to a Runtime.

        Tries exact match first, then prefix match.
        """
        if resolver in self._runtimes:
            return self._runtimes[resolver]
        prefix = resolver.split(".")[0]
        if prefix in self._runtimes:
            return self._runtimes[prefix]
        raise ResolutionError(
            f"No Runtime registered for resolver '{resolver}'"
        )

    async def resolve(  # noqa: D401
        self, task: Ask, feature: Feature, **kwargs: Any
    ) -> Any:
        """Dispatch an Ask to the appropriate runtime.

        The runner handles context resolution (a framework concern) and
        passes the Ask through to the runtime.  The runtime decides how
        to use the Ask's fields (prompt, input, input_type, output_type,
        to_prompt(), etc.).
        """
        actor = task.actor
        if not isinstance(actor, (AgentActor, InteractionActor)):
            raise ResolutionError(f"Unknown actor type: {type(actor).__name__}")
        runtime = self._resolve_runtime(actor.resolver)

        # Context resolution — framework concern
        context = ""
        if not task.continuation:
            all_keys: list[str] = []
            if isinstance(actor, AgentActor):
                all_keys = list(
                    dict.fromkeys(actor.context_keys + (task.context_keys or []))
                )
            elif task.context_keys:
                all_keys = list(task.context_keys)
            if all_keys:
                context = await self.context_provider.resolve(
                    all_keys, feature=feature
                )

        # Build runtime kwargs — framework-level metadata takes precedence
        # over user-supplied kwargs from runner.run(**kwargs)
        framework_kwargs: dict[str, Any] = {"context": context}
        if isinstance(actor, AgentActor):
            framework_kwargs["workspace"] = self.get_workspace(feature.workspace_id)
            framework_kwargs["session_key"] = f"{actor.name}:{feature.id}"

        merged = {**kwargs, **framework_kwargs}
        return await runtime.ask(task, **merged)

    def get_workspace(self, workspace_id: str | None) -> Workspace | None:
        if workspace_id is None:
            return None
        return self._workspaces.get(workspace_id)
