"""End-to-end test: two-phase workflow with mock runtimes."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    Ask,
    DefaultWorkflowRunner,
    Feature,
    Gate,
    InMemoryStore,
    InteractionActor,
    Phase,
    Role,
    Workflow,
    Workspace,
)
from iriai_compose.runtimes import AutoApproveRuntime
from tests.conftest import MockAgentRuntime


class E2EState(BaseModel):
    description: str = ""
    result: str = ""
    reviewed: bool = False


class AnalysisPhase(Phase):
    name = "analysis"

    async def execute(self, runner, feature, state):
        role = Role(name="analyst", prompt="You analyze things.")
        analyst = AgentActor(name="analyst", role=role, context_keys=["project"])
        result = await runner.run(
            Ask(actor=analyst, prompt=f"Analyze: {state.description}"),
            feature,
        )
        await runner.stores["artifacts"].put("analysis", str(result), feature=feature)
        state.result = str(result)
        return state


class ReviewPhase(Phase):
    name = "review"

    async def execute(self, runner, feature, state):
        human = InteractionActor(name="auto-reviewer", resolver="auto")
        approved = await runner.run(
            Gate(approver=human, prompt=f"Approve analysis: {state.result}?"),
            feature,
        )
        state.reviewed = approved is True
        return state


class E2EPipeline(Workflow):
    name = "e2e-pipeline"

    def build_phases(self):
        return [AnalysisPhase, ReviewPhase]


async def test_e2e_workflow():
    store = InMemoryStore()
    workspace = Workspace(id="main", path=Path("/tmp/e2e"), branch="main")

    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(response="analysis complete"),
            "auto": AutoApproveRuntime(),
        },
        stores={"artifacts": store},
        workspaces={"main": workspace},
    )

    feature = Feature(
        id="e2e-test",
        name="E2E Test",
        slug="e2e-test",
        workflow_name="e2e-pipeline",
        workspace_id="main",
    )

    state = E2EState(description="Test feature for e2e")
    result = await runner.execute_workflow(E2EPipeline(), feature, state)

    # Verify state flows through phases
    assert result.result == "analysis complete"
    assert result.reviewed is True

    # Verify artifacts stored
    stored = await store.get("analysis", feature=feature)
    assert stored == "analysis complete"


async def test_e2e_context_resolution():
    """Verify that context is resolved and included in prompts."""
    store = InMemoryStore()

    feature = Feature(
        id="ctx-test",
        name="Context Test",
        slug="ctx-test",
        workflow_name="test",
        workspace_id="main",
    )

    # Pre-populate store
    await store.put("project", "My project description", feature=feature)

    agent_rt = MockAgentRuntime(response="done")
    runner = DefaultWorkflowRunner(
        runtimes={"agent": agent_rt},
        stores={"artifacts": store},
    )

    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role, context_keys=["project"])
    await runner.run(Ask(actor=actor, prompt="Do work"), feature)

    # Verify context was included in the prompt
    prompt = agent_rt.calls[0]["prompt"]
    assert "My project description" in prompt
    assert "Do work" in prompt
