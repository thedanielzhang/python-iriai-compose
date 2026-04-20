from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from iriai_compose.prompts import Select
from iriai_compose.tasks import Ask, Task, to_str


def _gate_feedback_from_choice(choice: Any) -> str | None:
    if isinstance(choice, dict):
        feedback = choice.get("feedback")
        if isinstance(feedback, str):
            return feedback
        return None
    feedback = getattr(choice, "feedback", None)
    if isinstance(feedback, str):
        return feedback
    return None


class Interview(Task):
    """Example multi-turn interview task built from Ask."""

    questioner: Any
    responder: Any
    initial_prompt: str
    output_type: type[BaseModel] | None = None
    done: Callable[[Any], bool]

    async def execute(self, runner, feature, **kwargs: Any) -> Any:
        response = await runner.run(
            Ask(
                actor=self.questioner,
                prompt=self.initial_prompt,
                context_keys=self.context_keys,
                output_type=self.output_type,
            ),
            feature,
            **kwargs,
        )

        if self.done(response):
            return response

        while True:
            answer = await runner.run(
                Ask(actor=self.responder, prompt=to_str(response)),
                feature,
                **kwargs,
            )
            result = await runner.run(
                Ask(
                    actor=self.questioner,
                    prompt=f"The user responded:\n\n{to_str(answer)}",
                    context_keys=self.context_keys,
                    output_type=self.output_type,
                    continuation=True,
                ),
                feature,
                **kwargs,
            )
            if self.done(result):
                return result
            response = result


class Gate(Task):
    """Example approval gate built from Ask."""

    approver: Any
    prompt: str

    async def execute(self, runner, feature, **kwargs: Any) -> Any:
        choice = await runner.run(
            Ask(
                actor=self.approver,
                prompt=self.prompt,
                input=Select(options=["Approve", "Reject", "Give feedback"]),
                input_type=Select,
                context_keys=self.context_keys,
            ),
            feature,
            **kwargs,
        )
        feedback = _gate_feedback_from_choice(choice)
        if feedback is not None:
            feedback = feedback.strip()
            return feedback if feedback else False
        if choice == "Give feedback":
            return await runner.run(
                Ask(actor=self.approver, prompt="Please provide your feedback:"),
                feature,
                **kwargs,
            )
        return choice is True or choice == "Approve"


class Choose(Task):
    """Example selection task built from Ask."""

    chooser: Any
    prompt: str
    options: list[str]

    async def execute(self, runner, feature, **kwargs: Any) -> Any:
        return await runner.run(
            Ask(
                actor=self.chooser,
                prompt=self.prompt,
                input=Select(options=self.options),
                input_type=Select,
                context_keys=self.context_keys,
            ),
            feature,
            **kwargs,
        )


class Respond(Task):
    """Example free-form response task built from Ask."""

    responder: Any
    prompt: str

    async def execute(self, runner, feature, **kwargs: Any) -> Any:
        return await runner.run(
            Ask(
                actor=self.responder,
                prompt=self.prompt,
                context_keys=self.context_keys,
            ),
            feature,
            **kwargs,
        )
