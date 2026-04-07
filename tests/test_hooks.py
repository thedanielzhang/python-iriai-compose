"""Tests for on_start / on_done lifecycle hooks on Task, Phase, and Workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    Ask,
    DefaultContextProvider,
    DefaultWorkflowRunner,
    Feature,
    InMemoryArtifactStore,
    Phase,
    Role,
    Task,
    Workflow,
    Workspace,
)
from iriai_compose.exceptions import IriaiError, TaskExecutionError
from tests.conftest import MockAgentRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_role = Role(name="bot", prompt="You are a bot.")
_actor = AgentActor(name="bot", role=_role)


def _make_runner(response: str = "ok") -> DefaultWorkflowRunner:
    artifacts = InMemoryArtifactStore()
    return DefaultWorkflowRunner(
        runtimes={"agent": MockAgentRuntime(response=response)},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
    )


_feature = Feature(
    id="hook-test",
    name="Hook Test",
    slug="hook-test",
    workflow_name="test",
    workspace_id="main",
)


class HookState(BaseModel):
    value: str = ""


# ---------------------------------------------------------------------------
# Task hooks
# ---------------------------------------------------------------------------


class LoggingAsk(Ask):
    """Ask subclass that records hook calls."""

    log: list[str] = []

    async def on_start(self, runner, feature):
        self.log.append("on_start")

    async def on_done(self, runner, feature, *, result=None, error=None):
        if error:
            self.log.append(f"on_done:error:{type(error).__name__}")
        else:
            self.log.append(f"on_done:result:{result}")


async def test_task_hooks_on_success():
    runner = _make_runner(response="hello")
    task = LoggingAsk(actor=_actor, prompt="say hi", log=[])
    result = await runner.run(task, _feature)
    assert result == "hello"
    assert task.log == ["on_start", "on_done:result:hello"]


async def test_task_hooks_on_iriai_error():
    """on_done receives the IriaiError and re-raises it."""

    class FailTask(Task):
        log: list[str] = []

        async def on_start(self, runner, feature):
            self.log.append("on_start")

        async def on_done(self, runner, feature, *, result=None, error=None):
            self.log.append(f"on_done:error:{type(error).__name__}")

        async def execute(self, runner, feature, **kwargs):
            raise IriaiError("boom")

    task = FailTask(log=[])
    runner = _make_runner()
    with pytest.raises(IriaiError, match="boom"):
        await runner.run(task, _feature)
    assert task.log == ["on_start", "on_done:error:IriaiError"]


async def test_task_hooks_on_unexpected_error():
    """Unexpected exceptions are wrapped in TaskExecutionError."""

    class BadTask(Task):
        log: list[str] = []

        async def on_start(self, runner, feature):
            self.log.append("on_start")

        async def on_done(self, runner, feature, *, result=None, error=None):
            self.log.append(f"on_done:error:{type(error).__name__}")

        async def execute(self, runner, feature, **kwargs):
            raise RuntimeError("unexpected")

    task = BadTask(log=[])
    runner = _make_runner()
    with pytest.raises(TaskExecutionError):
        await runner.run(task, _feature)
    assert task.log == ["on_start", "on_done:error:TaskExecutionError"]


async def test_default_task_hooks_are_noop():
    """Default no-op hooks don't break anything."""
    runner = _make_runner(response="ok")
    task = Ask(actor=_actor, prompt="test")
    result = await runner.run(task, _feature)
    assert result == "ok"


# ---------------------------------------------------------------------------
# Phase hooks
# ---------------------------------------------------------------------------


class LoggingPhase(Phase):
    name = "logging-phase"

    def __init__(self, log: list[str]):
        self._log = log

    async def on_start(self, runner, feature, state):
        self._log.append(f"phase:on_start:{state.value}")

    async def on_done(self, runner, feature, state):
        self._log.append(f"phase:on_done:{state.value}")

    async def execute(self, runner, feature, state):
        state = state.model_copy(update={"value": "updated"})
        return state


class FailPhase(Phase):
    name = "fail-phase"

    def __init__(self, log: list[str]):
        self._log = log

    async def on_start(self, runner, feature, state):
        self._log.append("phase:on_start")

    async def on_done(self, runner, feature, state):
        self._log.append("phase:on_done")

    async def execute(self, runner, feature, state):
        raise RuntimeError("phase failed")


# ---------------------------------------------------------------------------
# Workflow hooks
# ---------------------------------------------------------------------------


class LoggingWorkflow(Workflow):
    name = "logging-workflow"

    def __init__(self, log: list[str], phases: list[type[Phase]] | None = None):
        self._log = log
        self._phases = phases or []

    async def on_start(self, runner, feature, state):
        self._log.append(f"workflow:on_start:{state.value}")

    async def on_done(self, runner, feature, state):
        self._log.append(f"workflow:on_done:{state.value}")

    def build_phases(self):
        return self._phases


async def test_workflow_and_phase_hooks_order():
    """Hooks fire in order: workflow.on_start, phase.on_start, phase.on_done, workflow.on_done."""
    log: list[str] = []

    # We need a phase class that captures the shared log.
    # Use a factory to create the class with access to the log list.
    class TrackingPhase(Phase):
        name = "tracking"

        async def on_start(self, runner, feature, state):
            log.append(f"phase:on_start:{state.value}")

        async def on_done(self, runner, feature, state):
            log.append(f"phase:on_done:{state.value}")

        async def execute(self, runner, feature, state):
            return state.model_copy(update={"value": "done"})

    workflow = LoggingWorkflow(log=log, phases=[TrackingPhase])
    runner = _make_runner()
    state = HookState(value="initial")
    result = await runner.execute_workflow(workflow, _feature, state)

    assert result.value == "done"
    assert log == [
        "workflow:on_start:initial",
        "phase:on_start:initial",
        "phase:on_done:done",
        "workflow:on_done:done",
    ]


async def test_phase_hooks_on_error():
    """Phase on_done is called even when execute raises, then workflow on_done fires."""
    log: list[str] = []

    class ErrorPhase(Phase):
        name = "error"

        async def on_start(self, runner, feature, state):
            log.append("phase:on_start")

        async def on_done(self, runner, feature, state):
            log.append("phase:on_done")

        async def execute(self, runner, feature, state):
            raise RuntimeError("boom")

    workflow = LoggingWorkflow(log=log, phases=[ErrorPhase])
    runner = _make_runner()

    with pytest.raises(RuntimeError, match="boom"):
        await runner.execute_workflow(workflow, _feature, HookState())

    assert log == [
        "workflow:on_start:",
        "phase:on_start",
        "phase:on_done",
        "workflow:on_done:",
    ]


async def test_default_phase_and_workflow_hooks_are_noop():
    """Default no-op hooks don't break the e2e path."""

    class SimplePhase(Phase):
        name = "simple"

        async def execute(self, runner, feature, state):
            return state.model_copy(update={"value": "simple"})

    class SimpleWorkflow(Workflow):
        name = "simple"

        def build_phases(self):
            return [SimplePhase]

    runner = _make_runner()
    result = await runner.execute_workflow(
        SimpleWorkflow(), _feature, HookState()
    )
    assert result.value == "simple"


async def test_multi_phase_hooks_order():
    """With two phases, hooks interleave correctly."""
    log: list[str] = []

    class PhaseA(Phase):
        name = "a"

        async def on_start(self, runner, feature, state):
            log.append("a:on_start")

        async def on_done(self, runner, feature, state):
            log.append("a:on_done")

        async def execute(self, runner, feature, state):
            return state.model_copy(update={"value": "a"})

    class PhaseB(Phase):
        name = "b"

        async def on_start(self, runner, feature, state):
            log.append("b:on_start")

        async def on_done(self, runner, feature, state):
            log.append("b:on_done")

        async def execute(self, runner, feature, state):
            return state.model_copy(update={"value": "b"})

    workflow = LoggingWorkflow(log=log, phases=[PhaseA, PhaseB])
    runner = _make_runner()
    result = await runner.execute_workflow(workflow, _feature, HookState())

    assert result.value == "b"
    assert log == [
        "workflow:on_start:",
        "a:on_start",
        "a:on_done",
        "b:on_start",
        "b:on_done",
        "workflow:on_done:b",
    ]


# ---------------------------------------------------------------------------
# Parallel tasks trigger hooks via run()
# ---------------------------------------------------------------------------


async def test_parallel_tasks_trigger_hooks():
    """parallel() routes through run(), so hooks fire for each task."""

    class TrackedAsk(Ask):
        log: list[str] = []

        async def on_start(self, runner, feature):
            self.log.append(f"on_start:{self.prompt}")

        async def on_done(self, runner, feature, *, result=None, error=None):
            self.log.append(f"on_done:{self.prompt}")

    role2 = Role(name="bot2", prompt="bot2")
    actor2 = AgentActor(name="bot2", role=role2)

    task_a = TrackedAsk(actor=_actor, prompt="a")
    task_b = TrackedAsk(actor=actor2, prompt="b")

    runner = _make_runner(response="done")
    results = await runner.parallel([task_a, task_b], _feature)

    assert results == ["done", "done"]
    # Each task's own log should record both hooks
    assert task_a.log == ["on_start:a", "on_done:a"]
    assert task_b.log == ["on_start:b", "on_done:b"]
