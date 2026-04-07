from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    AgentRuntime,
    Feature,
    InteractionActor,
    InteractionRuntime,
    Role,
    Workspace,
)
from iriai_compose.prompts import Select
from iriai_compose.tasks import Ask


class MockAgentRuntime(AgentRuntime):
    """Configurable: returns canned responses or calls a handler function."""

    name = "mock"

    def __init__(
        self,
        response: str | BaseModel | None = None,
        handler: Any = None,
    ) -> None:
        self._response = response or "mock response"
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def ask(self, task: Ask, **kwargs: Any) -> str | BaseModel:
        # Build prompt the same way a real agent runtime would
        context = kwargs.get("context", "")
        combined = task.to_prompt()
        prompt = (
            f"{context}\n\n## Task\n{combined}" if context else combined
        )
        call = {
            "prompt": prompt,
            "role": task.actor.role if isinstance(task.actor, AgentActor) else None,
            "output_type": task.output_type,
            "workspace": kwargs.get("workspace"),
            "session_key": kwargs.get("session_key"),
            "input": task.input,
            "input_type": task.input_type,
        }
        self.calls.append(call)
        if self._handler:
            return self._handler(call)
        return self._response


class MockInteractionRuntime(InteractionRuntime):
    """Returns canned responses based on input type."""

    name = "mock"

    def __init__(
        self,
        choose: str = "",
        respond: str = "mock input",
    ) -> None:
        self._choose = choose
        self._respond = respond
        self.calls: list[dict[str, Any]] = []

    async def ask(self, task: Ask, **kwargs: Any) -> str | bool:
        call = {
            "prompt": task.prompt,
            "input": task.input,
            "input_type": task.input_type,
            "output_type": task.output_type,
        }
        self.calls.append(call)
        if isinstance(task.input, Select):
            if self._choose:
                return self._choose
            return task.input.options[0] if task.input.options else ""
        return self._respond


@pytest.fixture
def pm_role() -> Role:
    return Role(
        name="pm",
        prompt="You are a PM.",
        tools=["Read", "Glob"],
    )


@pytest.fixture
def architect_role() -> Role:
    return Role(
        name="architect",
        prompt="You are an architect.",
        tools=["Read", "Bash"],
        model="claude-opus-4-6",
    )


@pytest.fixture
def agent_actor(pm_role: Role) -> AgentActor:
    return AgentActor(
        name="pm",
        role=pm_role,
        context_keys=["project"],
    )


@pytest.fixture
def interaction_actor() -> InteractionActor:
    return InteractionActor(name="user", resolver="human.slack")


@pytest.fixture
def feature() -> Feature:
    return Feature(
        id="test-feature",
        name="Test Feature",
        slug="test-feature",
        workflow_name="test",
        workspace_id="main",
    )


@pytest.fixture
def workspace() -> Workspace:
    return Workspace(
        id="main",
        path=Path("/tmp/test-workspace"),
        branch="main",
    )
