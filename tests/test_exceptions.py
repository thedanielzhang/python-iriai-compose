from iriai_compose import (
    AgentActor,
    InteractionActor,
    IriaiError,
    ResolutionError,
    Role,
    TaskExecutionError,
)
from examples._composite_tasks import Gate
from iriai_compose.tasks import Ask


def test_iriai_error_hierarchy():
    assert issubclass(ResolutionError, IriaiError)
    assert issubclass(TaskExecutionError, IriaiError)
    assert issubclass(IriaiError, Exception)


def test_resolution_error():
    err = ResolutionError("no runtime")
    assert str(err) == "no runtime"


def test_task_execution_error_with_ask():
    role = Role(name="pm", prompt="You are a PM.")
    actor = AgentActor(name="pm", role=role)
    from iriai_compose import Feature

    feature = Feature(
        id="f1", name="F1", slug="f1", workflow_name="test", workspace_id="main"
    )
    task = Ask(actor=actor, prompt="do something")
    err = TaskExecutionError(task=task, feature=feature, phase_name="planning")
    assert "Ask" in str(err)
    assert "planning" in str(err)
    assert "f1" in str(err)
    assert "pm" in str(err)


def test_task_execution_error_with_gate():
    human = InteractionActor(name="user", resolver="human")
    from iriai_compose import Feature

    feature = Feature(
        id="f2", name="F2", slug="f2", workflow_name="test", workspace_id="main"
    )
    task = Gate(approver=human, prompt="approve?")
    err = TaskExecutionError(task=task, feature=feature, phase_name="review")
    assert "user" in str(err)
    assert "Gate" in str(err)


def test_task_execution_error_unknown_actors():
    """Task with no recognized actor fields returns 'unknown'."""
    from iriai_compose.tasks import Task
    from iriai_compose import Feature

    class CustomTask(Task):
        data: str = "test"

        async def execute(self, runner, feature):
            pass

    feature = Feature(
        id="f3", name="F3", slug="f3", workflow_name="test", workspace_id="main"
    )
    task = CustomTask()
    err = TaskExecutionError(task=task, feature=feature, phase_name="custom")
    assert "unknown" in str(err)
