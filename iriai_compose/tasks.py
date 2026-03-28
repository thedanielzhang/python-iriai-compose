from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from iriai_compose.actors import Actor

if TYPE_CHECKING:
    from iriai_compose.runner import WorkflowRunner
    from iriai_compose.workflow import Feature


def to_str(value: Any) -> str:
    """Convert a value to a prompt-friendly string.

    Pydantic models are serialized as JSON; everything else uses str().
    """
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return str(value)


class Task(BaseModel, ABC):
    """Base class for all task types."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context_keys: list[str] = Field(default_factory=list)

    async def on_start(self, runner: WorkflowRunner, feature: Feature) -> None:
        """Called before execute. Override for setup."""

    async def on_done(
        self,
        runner: WorkflowRunner,
        feature: Feature,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """Called after execute. Override for teardown. error is set on failure."""

    @abstractmethod
    async def execute(self, runner: WorkflowRunner, feature: Feature) -> Any: ...


class Ask(Task):
    """One-shot: send prompt to one actor, get result."""

    actor: Actor
    prompt: str
    output_type: type[BaseModel] | None = None

    async def execute(self, runner: WorkflowRunner, feature: Feature) -> Any:
        return await runner.resolve(
            self.actor,
            self.prompt,
            feature=feature,
            context_keys=self.context_keys,
            output_type=self.output_type,
        )


class Interview(Task):
    """Multi-turn: questioner asks, responder answers, loop until done."""

    questioner: Actor
    responder: Actor
    initial_prompt: str
    output_type: type[BaseModel] | None = None
    done: Callable[[Any], bool]

    async def execute(self, runner: WorkflowRunner, feature: Feature) -> Any:
        response = await runner.resolve(
            self.questioner,
            self.initial_prompt,
            feature=feature,
            context_keys=self.context_keys,
            output_type=self.output_type,
        )

        if self.done(response):
            return response

        while True:
            answer = await runner.resolve(
                self.responder,
                to_str(response),
                feature=feature,
            )
            result = await runner.resolve(
                self.questioner,
                f"The user responded:\n\n{to_str(answer)}",
                feature=feature,
                context_keys=self.context_keys,
                output_type=self.output_type,
                continuation=True,
            )
            if self.done(result):
                return result
            response = result


class Gate(Task):
    """Approval: one actor approves, rejects, or gives feedback."""

    approver: Actor
    prompt: str

    async def execute(self, runner: WorkflowRunner, feature: Feature) -> Any:
        return await runner.resolve(
            self.approver,
            self.prompt,
            feature=feature,
            kind="approve",
        )


class Choose(Task):
    """Selection: one actor picks from options."""

    chooser: Actor
    prompt: str
    options: list[str]

    async def execute(self, runner: WorkflowRunner, feature: Feature) -> Any:
        return await runner.resolve(
            self.chooser,
            self.prompt,
            feature=feature,
            kind="choose",
            options=self.options,
        )


class Respond(Task):
    """Free-form: one actor provides open-ended input."""

    responder: Actor
    prompt: str

    async def execute(self, runner: WorkflowRunner, feature: Feature) -> Any:
        return await runner.resolve(
            self.responder,
            self.prompt,
            feature=feature,
            kind="respond",
        )
