from pydantic import BaseModel

from iriai_compose import AgentActor, InteractionActor, Role
from iriai_compose.prompts import Select
from iriai_compose.tasks import Ask, Choose, Gate, Interview, Respond


def test_ask_construction():
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    task = Ask(actor=actor, prompt="Do something")
    assert task.prompt == "Do something"
    assert task.output_type is None
    assert task.input is None
    assert task.input_type is None
    assert task.context_keys == []


def test_ask_with_output_type():
    class PRD(BaseModel):
        content: str

    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    task = Ask(actor=actor, prompt="Write PRD", output_type=PRD)
    assert task.output_type is PRD


def test_ask_with_context_keys():
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    task = Ask(actor=actor, prompt="Do something", context_keys=["threat-model"])
    assert task.context_keys == ["threat-model"]


def test_ask_with_input():
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    data = Select(options=["A", "B"])
    task = Ask(actor=actor, prompt="Pick one", input=data, input_type=Select)
    assert task.input is data
    assert task.input_type is Select


def test_ask_to_prompt_no_input():
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    task = Ask(actor=actor, prompt="Do something")
    assert task.to_prompt() == "Do something"


def test_ask_to_prompt_with_pydantic_input():
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    task = Ask(
        actor=actor,
        prompt="Pick one",
        input=Select(options=["A", "B"]),
    )
    result = task.to_prompt()
    assert "Pick one" in result
    assert '"options"' in result
    assert '"A"' in result


def test_ask_to_prompt_with_string_input():
    role = Role(name="pm", prompt="PM")
    actor = AgentActor(name="pm", role=role)
    task = Ask(actor=actor, prompt="Review this", input="some code here")
    result = task.to_prompt()
    assert result == "Review this\n\nsome code here"


def test_interview_construction():
    role = Role(name="pm", prompt="PM")
    questioner = AgentActor(name="pm", role=role)
    responder = InteractionActor(name="user", resolver="human")
    task = Interview(
        questioner=questioner,
        responder=responder,
        initial_prompt="Questions?",
        done=lambda r: True,
    )
    assert task.initial_prompt == "Questions?"
    assert task.done("anything") is True


def test_interview_done_predicate():
    role = Role(name="pm", prompt="PM")
    questioner = AgentActor(name="pm", role=role)
    responder = InteractionActor(name="user", resolver="human")
    task = Interview(
        questioner=questioner,
        responder=responder,
        initial_prompt="Questions?",
        done=lambda r: isinstance(r, str) and r == "DONE",
    )
    assert task.done("DONE") is True
    assert task.done("not done") is False


def test_gate_construction():
    human = InteractionActor(name="user", resolver="human")
    task = Gate(approver=human, prompt="Approve?")
    assert task.prompt == "Approve?"


def test_choose_construction():
    human = InteractionActor(name="user", resolver="human")
    task = Choose(chooser=human, prompt="Pick one", options=["A", "B", "C"])
    assert task.options == ["A", "B", "C"]


def test_respond_construction():
    human = InteractionActor(name="user", resolver="human")
    task = Respond(responder=human, prompt="Tell me more")
    assert task.prompt == "Tell me more"
