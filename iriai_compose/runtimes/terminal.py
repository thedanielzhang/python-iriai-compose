from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from iriai_compose.prompts import Confirm, Select
from iriai_compose.runner import InteractionRuntime

if TYPE_CHECKING:
    from iriai_compose.tasks import Ask


def _display_prompt(prompt: str) -> None:
    """Parse and display the prompt (which may be JSON with question/options)."""
    try:
        data = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        print(f"\n{prompt}")
        return

    if not isinstance(data, dict):
        print(f"\n{prompt}")
        return

    question = data.get("question")
    if not question:
        print(f"\n{prompt}")
        return

    print(f"\n{question}")
    options = data.get("options")
    if options:
        print()
        for i, opt in enumerate(options):
            print(f"  {i + 1}. {opt}")


def _ask_choose(prompt: str, options: list[str]) -> str:
    import questionary

    _display_prompt(prompt)
    print()
    return questionary.select("", choices=options).ask()


def _ask_confirm(prompt: str) -> bool:
    import questionary

    _display_prompt(prompt)
    print()
    return questionary.confirm("").ask()


def _ask_respond(prompt: str) -> str:
    import questionary

    _display_prompt(prompt)
    print()
    return questionary.text("").ask()


class TerminalInteractionRuntime(InteractionRuntime):
    """Interactive terminal-based interaction runtime.

    Uses fully-deferred imports — the class can be instantiated without
    questionary installed.  The dependency is imported lazily inside the
    ``_ask_*`` helpers, so an ``ImportError`` surfaces only when
    ``ask()`` is actually invoked without the package present.
    """

    name = "terminal"

    def __init__(self) -> None:
        pass

    async def ask(self, task: Ask, **kwargs: Any) -> str | bool:
        if isinstance(task.input, Select):
            return await asyncio.to_thread(
                _ask_choose, task.prompt, task.input.options
            )
        elif isinstance(task.input, Confirm):
            return await asyncio.to_thread(_ask_confirm, task.prompt)
        else:
            return await asyncio.to_thread(_ask_respond, task.prompt)
