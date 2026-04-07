import asyncio
import warnings

import pytest
from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    AgentRuntime,
    DefaultWorkflowRunner,
    Feature,
    InteractionActor,
    InteractionRuntime,
    InMemoryArtifactStore,
    DefaultContextProvider,
    ResolutionError,
    Role,
    TaskExecutionError,
    Workspace,
    Phase,
    Workflow,
)
from iriai_compose.prompts import Select
from iriai_compose.tasks import Ask, Gate, Interview, Respond, Choose, Task
from tests.conftest import MockAgentRuntime, MockInteractionRuntime


@pytest.fixture
def feature():
    return Feature(
        id="f1", name="F1", slug="f1", workflow_name="test", workspace_id="main"
    )


@pytest.fixture
def workspace():
    from pathlib import Path

    return Workspace(id="main", path=Path("/tmp/ws"), branch="main")


@pytest.fixture
def artifacts():
    return InMemoryArtifactStore()


@pytest.fixture
def runner(artifacts, workspace):
    return DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(response="agent response"),
            "human": MockInteractionRuntime(),
            "auto": MockInteractionRuntime(respond="auto"),
        },
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )


# --- Deprecated constructor ---


def test_deprecated_constructor(artifacts, workspace):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        runner = DefaultWorkflowRunner(
            agent_runtime=MockAgentRuntime(response="ok"),
            interaction_runtimes={"human": MockInteractionRuntime()},
            artifacts=artifacts,
            context_provider=DefaultContextProvider(artifacts=artifacts),
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
    # Runtime should still be accessible
    assert runner._runtimes["agent"] is not None
    assert runner._runtimes["human"] is not None


# --- Runtime routing ---


def test_resolve_runtime_exact(runner):
    rt = runner._resolve_runtime("human")
    assert rt is not None


def test_resolve_runtime_prefix(runner):
    rt = runner._resolve_runtime("human.slack")
    assert rt is runner._runtimes["human"]


def test_resolve_runtime_miss(runner):
    with pytest.raises(ResolutionError):
        runner._resolve_runtime("unknown.runtime")


# --- Context merging ---


async def test_context_merging(runner, feature, artifacts):
    await artifacts.put("project", "Project info", feature=feature)
    await artifacts.put("extra", "Extra info", feature=feature)

    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role, context_keys=["project"])
    task = Ask(actor=actor, prompt="Do something", context_keys=["extra"])

    await runner.run(task, feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    prompt = agent_rt.calls[-1]["prompt"]
    assert "Project info" in prompt
    assert "Extra info" in prompt


async def test_context_dedup(runner, feature, artifacts):
    await artifacts.put("project", "Project info", feature=feature)

    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role, context_keys=["project"])
    task = Ask(actor=actor, prompt="Do something", context_keys=["project"])

    await runner.run(task, feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    prompt = agent_rt.calls[-1]["prompt"]
    # "project" should appear once in context, not twice
    assert prompt.count("## project") == 1


async def test_no_context_keys_no_prefix(runner, feature):
    """When actor has no context_keys, prompt is passed through without prefix."""
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role, context_keys=[])
    await runner.run(Ask(actor=actor, prompt="raw prompt"), feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    assert agent_rt.calls[-1]["prompt"] == "raw prompt"


# --- Session key derivation ---


async def test_session_key(runner, feature):
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    await runner.run(Ask(actor=actor, prompt="test"), feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    assert agent_rt.calls[-1]["session_key"] == "pm:f1"


# --- Workspace passed to runtime ---


async def test_workspace_passed_to_agent_runtime(runner, feature, workspace):
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    await runner.run(Ask(actor=actor, prompt="test"), feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    assert agent_rt.calls[-1]["workspace"] is workspace


# --- Error wrapping ---


async def test_non_iriai_error_wrapped(runner, feature):
    class FailingTask(Task):
        async def execute(self, runner, feature, **kwargs):
            raise ValueError("boom")

    with pytest.raises(TaskExecutionError) as exc_info:
        await runner.run(FailingTask(), feature)
    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_iriai_error_passthrough(runner, feature):
    class FailingTask(Task):
        async def execute(self, runner, feature, **kwargs):
            raise ResolutionError("no runtime")

    with pytest.raises(ResolutionError):
        await runner.run(FailingTask(), feature)


async def test_error_wrapping_preserves_phase_name(runner, feature):
    class FailingTask(Task):
        async def execute(self, runner, feature, **kwargs):
            raise ValueError("boom")

    with pytest.raises(TaskExecutionError) as exc_info:
        await runner.run(FailingTask(), feature, phase_name="my-phase")
    assert "my-phase" in str(exc_info.value)


# --- Resolve unknown actor ---


async def test_resolve_unknown_actor(runner, feature):
    from iriai_compose import Actor

    ask = Ask(actor=Actor(name="plain"), prompt="test")
    with pytest.raises(ResolutionError):
        await runner.resolve(ask, feature)


# --- Parallel ---


async def test_parallel_success(runner, feature):
    role = Role(name="pm", prompt="PM")
    actor_a = AgentActor(name="pm-a", role=role)
    actor_b = AgentActor(name="pm-b", role=role)
    tasks = [
        Ask(actor=actor_a, prompt="task1"),
        Ask(actor=actor_b, prompt="task2"),
    ]
    results = await runner.parallel(tasks, feature)
    assert len(results) == 2
    assert all(r == "agent response" for r in results)


async def test_parallel_preserves_order(runner, feature):
    responses = iter(["first", "second", "third"])

    def handler(call):
        return next(responses)

    runner_ordered = DefaultWorkflowRunner(
        runtimes={"agent": MockAgentRuntime(handler=handler)},
        artifacts=runner.artifacts,
        context_provider=runner.context_provider,
    )
    role = Role(name="pm", prompt="PM")
    actors = [AgentActor(name=f"pm-{i}", role=role) for i in range(3)]
    tasks = [
        Ask(actor=actors[0], prompt="a"),
        Ask(actor=actors[1], prompt="b"),
        Ask(actor=actors[2], prompt="c"),
    ]
    results = await runner_ordered.parallel(tasks, feature)
    assert results == ["first", "second", "third"]


async def test_parallel_fail_fast_cancels(feature, artifacts, workspace):
    started = []
    finished = []

    class SlowTask(Task):
        label: str = ""

        async def execute(self, runner, feature, **kwargs):
            started.append(self.label)
            await asyncio.sleep(10)
            finished.append(self.label)
            return "done"

    class FailTask(Task):
        async def execute(self, runner, feature, **kwargs):
            raise ValueError("boom")

    runner = DefaultWorkflowRunner(
        runtimes={"agent": MockAgentRuntime()},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )
    with pytest.raises(ExceptionGroup):
        await runner.parallel(
            [FailTask(), SlowTask(label="slow")], feature, fail_fast=True
        )
    assert "slow" not in finished


async def test_parallel_fail_fast(feature, artifacts, workspace):
    def handler(call):
        if "fail" in call["prompt"]:
            raise ValueError("boom")
        return "ok"

    runner = DefaultWorkflowRunner(
        runtimes={"agent": MockAgentRuntime(handler=handler)},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )
    role = Role(name="pm", prompt="PM")
    actor_a = AgentActor(name="pm-a", role=role)
    actor_b = AgentActor(name="pm-b", role=role)
    tasks = [
        Ask(actor=actor_a, prompt="fail"),
        Ask(actor=actor_b, prompt="ok"),
    ]
    with pytest.raises(ExceptionGroup):
        await runner.parallel(tasks, feature, fail_fast=True)


async def test_parallel_no_fail_fast(feature, artifacts, workspace):
    def handler(call):
        if "fail" in call["prompt"]:
            raise ValueError("boom")
        return "ok"

    runner = DefaultWorkflowRunner(
        runtimes={"agent": MockAgentRuntime(handler=handler)},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )
    role = Role(name="pm", prompt="PM")
    actor_a = AgentActor(name="pm-a", role=role)
    actor_b = AgentActor(name="pm-b", role=role)
    tasks = [
        Ask(actor=actor_a, prompt="fail"),
        Ask(actor=actor_b, prompt="succeed"),
    ]
    with pytest.raises(ExceptionGroup):
        await runner.parallel(tasks, feature, fail_fast=False)


async def test_parallel_rejects_duplicate_actors(runner, feature):
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    tasks = [
        Ask(actor=actor, prompt="task1"),
        Ask(actor=actor, prompt="task2"),
    ]
    with pytest.raises(ValueError, match="Actor 'pm' used in multiple parallel tasks"):
        await runner.parallel(tasks, feature)


async def test_parallel_empty(runner, feature):
    results = await runner.parallel([], feature)
    assert results == []


# --- Workspace lookup ---


def test_get_workspace(runner, workspace):
    ws = runner.get_workspace("main")
    assert ws is workspace


def test_get_workspace_none(runner):
    assert runner.get_workspace(None) is None


def test_get_workspace_missing(runner):
    assert runner.get_workspace("nonexistent") is None


# --- Workflow execution ---


async def test_execute_workflow(runner, feature):
    class State(BaseModel):
        value: int = 0

    class PhaseA(Phase):
        name = "phase-a"

        async def execute(self, runner, feature, state):
            state.value += 1
            return state

    class PhaseB(Phase):
        name = "phase-b"

        async def execute(self, runner, feature, state):
            state.value += 10
            return state

    class TestWorkflow(Workflow):
        name = "test"

        def build_phases(self):
            return [PhaseA, PhaseB]

    result = await runner.execute_workflow(TestWorkflow(), feature, State())
    assert result.value == 11


async def test_execute_workflow_sets_phase(runner, feature):
    from iriai_compose.runner import _current_phase_var

    phase_names = []

    class State(BaseModel):
        pass

    class PhaseA(Phase):
        name = "alpha"

        async def execute(self, r, feature, state):
            phase_names.append(_current_phase_var.get())
            return state

    class PhaseB(Phase):
        name = "beta"

        async def execute(self, r, feature, state):
            phase_names.append(_current_phase_var.get())
            return state

    class TestWorkflow(Workflow):
        name = "test"

        def build_phases(self):
            return [PhaseA, PhaseB]

    await runner.execute_workflow(TestWorkflow(), feature, State())
    assert phase_names == ["alpha", "beta"]


async def test_execute_workflow_empty_phases(runner, feature):
    class State(BaseModel):
        value: int = 42

    class EmptyWorkflow(Workflow):
        name = "empty"

        def build_phases(self):
            return []

    result = await runner.execute_workflow(EmptyWorkflow(), feature, State())
    assert result.value == 42


# --- execute_child ---


async def test_execute_child(runner, feature):
    class State(BaseModel):
        value: int = 0

    class ChildPhase(Phase):
        name = "child"

        async def execute(self, runner, feature, state):
            state.value = 42
            return state

    class ChildWorkflow(Workflow):
        name = "child"

        def build_phases(self):
            return [ChildPhase]

    result = await runner.execute_child(ChildWorkflow(), feature, State())
    assert result.value == 42


async def test_execute_child_rebinds_workspace(runner, feature):
    captured_workspace_ids = []

    class State(BaseModel):
        pass

    class ChildPhase(Phase):
        name = "child"

        async def execute(self, runner, feat, state):
            captured_workspace_ids.append(feat.workspace_id)
            return state

    class ChildWorkflow(Workflow):
        name = "child"

        def build_phases(self):
            return [ChildPhase]

    await runner.execute_child(
        ChildWorkflow(), feature, State(), workspace_id="other"
    )
    assert captured_workspace_ids == ["other"]


async def test_execute_child_does_not_mutate_original_feature(runner, feature):
    original_ws = feature.workspace_id

    class State(BaseModel):
        pass

    class ChildPhase(Phase):
        name = "child"

        async def execute(self, runner, feat, state):
            return state

    class ChildWorkflow(Workflow):
        name = "child"

        def build_phases(self):
            return [ChildPhase]

    await runner.execute_child(
        ChildWorkflow(), feature, State(), workspace_id="other"
    )
    assert feature.workspace_id == original_ws


# --- Full Ask flow ---


async def test_ask_agent_flow(runner, feature):
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    result = await runner.run(
        Ask(actor=actor, prompt="Write something"), feature
    )
    assert result == "agent response"


# --- Gate flow ---


async def test_gate_approve_flow(runner, feature):
    human = InteractionActor(name="user", resolver="human")
    result = await runner.run(
        Gate(approver=human, prompt="Approve?"), feature
    )
    # MockInteractionRuntime returns first option ("Approve") for Select input
    assert result is True


async def test_gate_reject_flow(feature, artifacts, workspace):
    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(),
            "human": MockInteractionRuntime(choose="Reject"),
        },
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )
    human = InteractionActor(name="user", resolver="human")
    result = await runner.run(
        Gate(approver=human, prompt="Approve?"), feature
    )
    assert result is False


async def test_gate_feedback_flow(feature, artifacts, workspace):
    """Gate returns feedback string when 'Give feedback' is chosen."""
    call_count = 0

    class FeedbackRuntime(InteractionRuntime):
        name = "mock"

        async def ask(self, task, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Give feedback"
            return "needs more tests"

    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(),
            "human": FeedbackRuntime(),
        },
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )
    human = InteractionActor(name="user", resolver="human")
    result = await runner.run(
        Gate(approver=human, prompt="Approve?"), feature
    )
    assert result == "needs more tests"


# --- Choose flow ---


async def test_choose_flow(runner, feature):
    human = InteractionActor(name="user", resolver="human")
    result = await runner.run(
        Choose(chooser=human, prompt="Pick one", options=["A", "B", "C"]),
        feature,
    )
    # MockInteractionRuntime returns first option for Select
    assert result == "A"


# --- Respond flow ---


async def test_respond_flow(runner, feature):
    human = InteractionActor(name="user", resolver="human")
    mock_rt: MockInteractionRuntime = runner._runtimes["human"]
    mock_rt._respond = "user input"
    result = await runner.run(
        Respond(responder=human, prompt="Tell me more"), feature
    )
    assert result == "user input"


# --- Ask with input + to_prompt ---


async def test_ask_to_prompt_with_input(runner, feature):
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    result = await runner.run(
        Ask(
            actor=actor,
            prompt="Rank these",
            input=Select(options=["A", "B"]),
            input_type=Select,
        ),
        feature,
    )
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    prompt = agent_rt.calls[-1]["prompt"]
    assert "Rank these" in prompt
    assert '"options"' in prompt  # input appended as JSON


async def test_ask_to_prompt_no_input(runner, feature):
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    await runner.run(Ask(actor=actor, prompt="plain prompt"), feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    assert agent_rt.calls[-1]["prompt"] == "plain prompt"


# --- Interview flow ---


async def test_interview_flow(feature, artifacts, workspace):
    call_count = 0

    def handler(call):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            return "DONE"
        return "question"

    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(handler=handler),
            "human": MockInteractionRuntime(respond="my answer"),
        },
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )

    role = Role(name="pm", prompt="PM")
    questioner = AgentActor(name="pm", role=role)
    responder = InteractionActor(name="user", resolver="human")

    result = await runner.run(
        Interview(
            questioner=questioner,
            responder=responder,
            initial_prompt="What do you need?",
            done=lambda r: r == "DONE",
        ),
        feature,
    )
    assert result == "DONE"


async def test_interview_immediate_done(feature, artifacts, workspace):
    def handler(call):
        return "DONE"

    mock_interaction = MockInteractionRuntime(respond="answer")
    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(handler=handler),
            "human": mock_interaction,
        },
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )

    role = Role(name="pm", prompt="PM")
    questioner = AgentActor(name="pm", role=role)
    responder = InteractionActor(name="user", resolver="human")

    result = await runner.run(
        Interview(
            questioner=questioner,
            responder=responder,
            initial_prompt="Start",
            done=lambda r: r == "DONE",
        ),
        feature,
    )
    assert result == "DONE"
    assert len(mock_interaction.calls) == 0


async def test_interview_pydantic_model_serialization(feature, artifacts, workspace):
    class Result(BaseModel):
        status: str
        questions: list[str] = []

    call_count = 0

    def handler(call):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Result(status="asking", questions=["q1"])
        return Result(status="done", questions=[])

    mock_interaction = MockInteractionRuntime(respond="my answer")
    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MockAgentRuntime(handler=handler),
            "human": mock_interaction,
        },
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
    )

    role = Role(name="pm", prompt="PM")
    questioner = AgentActor(name="pm", role=role)
    responder = InteractionActor(name="user", resolver="human")

    result = await runner.run(
        Interview(
            questioner=questioner,
            responder=responder,
            initial_prompt="Start",
            done=lambda r: isinstance(r, Result) and not r.questions,
        ),
        feature,
    )
    assert isinstance(result, Result)
    assert result.status == "done"
    # The responder should have received JSON, not Python repr
    responder_prompt = mock_interaction.calls[0]["prompt"]
    assert '"status"' in responder_prompt
    assert "Result(" not in responder_prompt


async def test_interview_agent_to_agent(feature, artifacts, workspace):
    call_count = 0

    def handler(call):
        nonlocal call_count
        call_count += 1
        return "DONE"

    runner = DefaultWorkflowRunner(
        runtimes={"agent": MockAgentRuntime(handler=handler)},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
    )

    role_q = Role(name="questioner", prompt="Q")
    role_r = Role(name="responder", prompt="R")
    questioner = AgentActor(name="q", role=role_q, context_keys=["project"])
    responder = AgentActor(name="r", role=role_r, context_keys=["prd"])

    result = await runner.run(
        Interview(
            questioner=questioner,
            responder=responder,
            initial_prompt="Start",
            done=lambda r: r == "DONE",
        ),
        feature,
    )
    assert result == "DONE"


# --- Continuation skips context ---


async def test_continuation_skips_context(runner, feature, artifacts):
    """Ask with continuation=True should skip context injection."""
    await artifacts.put("project", "Project info", feature=feature)

    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role, context_keys=["project"])

    # First call — context injected
    await runner.run(Ask(actor=actor, prompt="initial"), feature)
    agent_rt: MockAgentRuntime = runner._runtimes["agent"]
    assert "Project info" in agent_rt.calls[-1]["prompt"]

    # Continuation — context skipped
    await runner.run(
        Ask(actor=actor, prompt="follow-up", continuation=True), feature
    )
    assert "Project info" not in agent_rt.calls[-1]["prompt"]
    assert "follow-up" in agent_rt.calls[-1]["prompt"]


# --- kwargs flow through to runtime ---


async def test_kwargs_flow_to_runtime(feature, artifacts, workspace):
    """Extra kwargs from runner.run() should reach the runtime."""
    received_kwargs = {}

    class KwargsCapture(AgentRuntime):
        name = "capture"

        async def ask(self, task, **kwargs):
            received_kwargs.update(kwargs)
            return "ok"

    runner = DefaultWorkflowRunner(
        runtimes={"agent": KwargsCapture()},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
        workspaces={"main": workspace},
    )
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)

    await runner.run(
        Ask(actor=actor, prompt="test"),
        feature,
        custom_hint="hello",
    )
    # Framework kwargs always present
    assert "context" in received_kwargs
    assert "workspace" in received_kwargs
    assert "session_key" in received_kwargs
    # User kwargs forwarded
    assert received_kwargs["custom_hint"] == "hello"
