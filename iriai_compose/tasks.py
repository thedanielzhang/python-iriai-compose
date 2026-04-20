from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
    async def execute(
        self, runner: WorkflowRunner, feature: Feature, **kwargs: Any
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Ask — the only leaf task (atomic)
# ---------------------------------------------------------------------------


class Ask(Task):
    """One-shot: send prompt to one actor, get result.

    This is the only task type that reaches the runtime directly via
    ``runner.resolve()``.  All other tasks compose Asks in their
    ``execute()`` method.
    """

    actor: Actor
    prompt: str
    input: Any = None
    input_type: type[BaseModel] | None = None
    output_type: type[BaseModel] | None = None
    continuation: bool = False

    def to_prompt(self) -> str:
        """Combine prompt + input into the final prompt string.

        Default: appends input as JSON.  Subclasses can override for
        custom templating (e.g. referencing specific input fields).
        """
        if self.input is None:
            return self.prompt
        if isinstance(self.input, BaseModel):
            return f"{self.prompt}\n\n{self.input.model_dump_json(indent=2)}"
        return f"{self.prompt}\n\n{self.input}"

    async def execute(
        self, runner: WorkflowRunner, feature: Feature, **kwargs: Any
    ) -> Any:
        return await runner.resolve(self, feature, **kwargs)
