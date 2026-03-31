from __future__ import annotations

import asyncio
import json

from iriai_compose.pending import Pending
from iriai_compose.runner import InteractionRuntime

_APPROVE_APPROVE = "Approve"
_APPROVE_REJECT = "Reject"
_APPROVE_FEEDBACK = "Give feedback"


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


def _ask_approve(prompt: str) -> bool | str:
    import questionary

    _display_prompt(prompt)
    print()
    choice = questionary.select(
        "",
        choices=[_APPROVE_APPROVE, _APPROVE_REJECT, _APPROVE_FEEDBACK],
    ).ask()
    if choice == _APPROVE_APPROVE:
        return True
    if choice == _APPROVE_REJECT:
        return False
    return questionary.text("Feedback:").ask()


def _ask_choose(prompt: str, options: list[str]) -> str:
    import questionary

    _display_prompt(prompt)
    print()
    return questionary.select("", choices=options).ask()


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
    ``resolve()`` is actually invoked without the package present.
    """

    name = "terminal"

    def __init__(self) -> None:
        # No eager import check — questionary is imported lazily inside the
        # _ask_* helpers so the class can be instantiated (and tested with
        # mocked callables) without the optional dependency installed.
        pass

    async def resolve(self, pending: Pending) -> str | bool:
        if pending.kind == "approve":
            return await asyncio.to_thread(_ask_approve, pending.prompt)
        elif pending.kind == "choose":
            options = pending.options or []
            return await asyncio.to_thread(_ask_choose, pending.prompt, options)
        else:  # respond
            return await asyncio.to_thread(_ask_respond, pending.prompt)
