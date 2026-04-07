from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iriai_compose.prompts import Confirm, Select
from iriai_compose.runner import InteractionRuntime
from iriai_compose.runtimes.terminal import TerminalInteractionRuntime

if TYPE_CHECKING:
    from iriai_compose.tasks import Ask


class AutoApproveRuntime(InteractionRuntime):
    """Auto-approves all interaction requests."""

    name = "auto"

    async def ask(self, task: Ask, **kwargs: Any) -> str | bool:
        if isinstance(task.input, Select):
            return task.input.options[0] if task.input.options else ""
        if isinstance(task.input, Confirm):
            return True
        return "auto-approved"


__all__ = ["TerminalInteractionRuntime", "AutoApproveRuntime"]
