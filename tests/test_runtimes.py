from unittest.mock import patch

import pytest

from iriai_compose import InteractionActor
from iriai_compose.prompts import Confirm, Select
from iriai_compose.runtimes import AutoApproveRuntime, TerminalInteractionRuntime
from iriai_compose.tasks import Ask

_actor = InteractionActor(name="user", resolver="test")


def _ask(prompt="test", **kwargs):
    """Helper to create an Ask for testing runtimes."""
    return Ask(actor=_actor, prompt=prompt, **kwargs)


# --- AutoApproveRuntime ---


async def test_auto_approve_select():
    rt = AutoApproveRuntime()
    result = await rt.ask(_ask("Pick one", input=Select(options=["A", "B"]), input_type=Select))
    assert result == "A"


async def test_auto_approve_confirm():
    rt = AutoApproveRuntime()
    result = await rt.ask(_ask("Are you sure?", input=Confirm(), input_type=Confirm))
    assert result is True


async def test_auto_approve_respond():
    rt = AutoApproveRuntime()
    result = await rt.ask(_ask("Tell me more"))
    assert result == "auto-approved"


# --- TerminalInteractionRuntime ---


async def test_terminal_select():
    rt = TerminalInteractionRuntime()
    with patch(
        "iriai_compose.runtimes.terminal.asyncio.to_thread", return_value="B"
    ):
        result = await rt.ask(
            _ask("Pick one", input=Select(options=["A", "B", "C"]), input_type=Select)
        )
    assert result == "B"


async def test_terminal_confirm():
    rt = TerminalInteractionRuntime()
    with patch(
        "iriai_compose.runtimes.terminal.asyncio.to_thread", return_value=True
    ):
        result = await rt.ask(_ask("Are you sure?", input=Confirm(), input_type=Confirm))
    assert result is True


async def test_terminal_respond():
    rt = TerminalInteractionRuntime()
    with patch(
        "iriai_compose.runtimes.terminal.asyncio.to_thread", return_value="my feedback"
    ):
        result = await rt.ask(_ask("Tell me more"))
    assert result == "my feedback"


# --- Deferred import errors ---


async def test_terminal_runtime_import_error():
    """Instantiation succeeds without questionary; ImportError surfaces lazily
    when ask() delegates to a helper that imports it."""
    rt = TerminalInteractionRuntime()  # should NOT raise
    with patch.dict("sys.modules", {"questionary": None}):
        with pytest.raises(ImportError, match="questionary"):
            await rt.ask(_ask("Tell me more"))


def test_claude_runtime_removed():
    """ClaudeAgentRuntime has been moved to iriai-build-v2."""
    import importlib

    assert importlib.util.find_spec("iriai_compose.runtimes.claude") is None
