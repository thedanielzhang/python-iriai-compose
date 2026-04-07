"""Claude-driven code review workflow.

Uses real ClaudeAgentRuntime (backed by claude-agent-sdk) to review
files in this repository. The reviewer agent can Read, Glob, and Grep.

Prerequisites:
  pip install claude-agent-sdk
  export ANTHROPIC_API_KEY=your-api-key

Run:  python examples/claude_code_review.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    Ask,
    Choose,
    DefaultWorkflowRunner,
    Feature,
    Gate,
    InMemorySessionStore,
    InMemoryStore,
    InteractionActor,
    Phase,
    Respond,
    Role,
    Workflow,
    Workspace,
)
from iriai_compose.runtimes import TerminalInteractionRuntime
from iriai_compose.runtimes.claude import ClaudeAgentRuntime


# ---------------------------------------------------------------------------
# Streaming display
# ---------------------------------------------------------------------------

def print_stream(msg) -> None:
    """Print agent messages to the terminal as they stream in."""
    from claude_agent_sdk.types import AssistantMessage

    if not isinstance(msg, AssistantMessage):
        return

    for block in msg.content:
        typ = type(block).__name__
        if typ == "TextBlock":
            print(block.text, end="", flush=True)
        elif typ == "ToolUseBlock":
            tool_input = block.input
            if isinstance(tool_input, dict):
                # Show a concise summary of the tool call
                target = (
                    tool_input.get("file_path")
                    or tool_input.get("pattern")
                    or tool_input.get("command", "")
                )
            else:
                target = str(tool_input)
            print(f"\n[tool] {block.name} {target}", flush=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReviewState(BaseModel):
    file_target: str = ""
    review: str = ""
    strategy: str = ""
    resolved: bool = False


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------

reviewer_role = Role(
    name="code-reviewer",
    prompt=(
        "You are a senior code reviewer. Analyze the code thoroughly for:\n"
        "- Correctness and potential bugs\n"
        "- Design patterns and architecture\n"
        "- Error handling gaps\n"
        "- Naming and readability\n"
        "Be specific — reference line numbers and function names."
    ),
    tools=["Read", "Glob", "Grep"],
    model="claude-sonnet-4-6",
)

reviewer = AgentActor(name="reviewer", role=reviewer_role)
human = InteractionActor(name="developer", resolver="terminal")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

class ReviewPhase(Phase):
    """Human picks a target, Claude reviews it."""

    name = "review"

    async def execute(self, runner, feature, state):
        # Human describes what to review
        target = await runner.run(
            Respond(
                responder=human,
                prompt="What file or area would you like reviewed? (e.g. iriai_compose/runner.py)",
            ),
            feature,
        )
        state.file_target = target

        # Claude reviews it
        review = await runner.run(
            Ask(
                actor=reviewer,
                prompt=(
                    f"Review the following code in this repository: {target}\n\n"
                    "Read the file(s), analyze them, and provide a detailed code review."
                ),
            ),
            feature,
        )
        state.review = str(review)
        await runner.stores["artifacts"].put("review", state.review, feature=feature)
        print(f"\n{'='*60}")  # visual separator after streamed review
        return state


class TriagePhase(Phase):
    """Human approves or requests deeper analysis."""

    name = "triage"

    async def execute(self, runner, feature, state):
        # Human approves or rejects
        approved = await runner.run(
            Gate(approver=human, prompt="Accept this review as-is?"),
            feature,
        )

        if approved is True:
            state.resolved = True
            return state

        # Human picks a follow-up strategy
        strategy = await runner.run(
            Choose(
                chooser=human,
                prompt="What should the reviewer focus on next?",
                options=[
                    "Deeper analysis — find subtle bugs and edge cases",
                    "Focus on tests — review test coverage and quality",
                    "Focus on types/contracts — check interfaces and type safety",
                    "Move on — accept current review",
                ],
            ),
            feature,
        )
        state.strategy = strategy

        if "Move on" in strategy:
            state.resolved = True
            return state

        # Claude does a follow-up pass
        followup = await runner.run(
            Ask(
                actor=reviewer,
                prompt=(
                    f"Do a follow-up review of {state.file_target} with this focus:\n"
                    f"{strategy}\n\n"
                    "Build on your previous review — don't repeat what you already covered."
                ),
                context_keys=["review"],
            ),
            feature,
        )
        print(f"\n{'='*60}")  # visual separator after streamed follow-up

        # Final confirmation
        confirmed = await runner.run(
            Gate(approver=human, prompt="Satisfied with the review?"),
            feature,
        )
        state.resolved = confirmed is True
        return state


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class ClaudeCodeReviewWorkflow(Workflow):
    name = "claude-code-review"

    def build_phases(self):
        return [ReviewPhase, TriagePhase]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    repo_root = Path(__file__).resolve().parent.parent

    store = InMemoryStore()
    sessions = InMemorySessionStore()
    workspace = Workspace(id="main", path=repo_root, branch="main")

    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": ClaudeAgentRuntime(session_store=sessions, on_message=print_stream),
            "terminal": TerminalInteractionRuntime(),
        },
        stores={"artifacts": store},
        workspaces={"main": workspace},
    )

    feature = Feature(
        id="cr-live",
        name="Live Code Review",
        slug="live-code-review",
        workflow_name="claude-code-review",
        workspace_id="main",
    )

    print("=== Claude Code Review Workflow ===\n")
    state = ReviewState()
    result = await runner.execute_workflow(ClaudeCodeReviewWorkflow(), feature, state)
    print(f"\nDone. Resolved: {result.resolved}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
