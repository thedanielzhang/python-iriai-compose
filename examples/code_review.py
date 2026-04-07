"""Example A: Simple linear code review workflow.

Two-phase workflow exercising Ask, Gate, Choose, Respond,
TerminalInteractionRuntime, artifacts, and context resolution.

Run interactively:   python examples/code_review.py
Run auto-approved:   python examples/code_review.py --auto
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    Ask,
    Choose,
    DefaultWorkflowRunner,
    Feature,
    Gate,
    InMemoryStore,
    InteractionActor,
    Phase,
    Respond,
    Role,
    Workflow,
    Workspace,
)
from iriai_compose.runtimes import AutoApproveRuntime, TerminalInteractionRuntime

from runtime import EchoAgentRuntime


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReviewState(BaseModel):
    description: str = ""
    review: str = ""
    strategy: str = ""
    resolved: bool = False


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------

reviewer_role = Role(
    name="code-reviewer",
    prompt="You are an experienced code reviewer. Be thorough but constructive.",
    tools=["Read", "Glob", "Grep"],
)

reviewer = AgentActor(name="reviewer", role=reviewer_role, context_keys=["submission"])
human = InteractionActor(name="developer", resolver="terminal")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

class SubmissionPhase(Phase):
    name = "submission"

    async def execute(self, runner, feature, state):
        # Human describes the code to review
        description = await runner.run(
            Respond(responder=human, prompt="Describe the code you want reviewed:"),
            feature,
        )
        state.description = description
        await runner.stores["artifacts"].put("submission", description, feature=feature)

        # Agent reviews it
        review = await runner.run(
            Ask(
                actor=reviewer,
                prompt=f"Review this code submission:\n\n{description}",
            ),
            feature,
        )
        state.review = review
        await runner.stores["artifacts"].put("review", review, feature=feature)
        print(f"\n--- Review ---\n{review}\n")

        # Human approves the review
        approved = await runner.run(
            Gate(approver=human, prompt="Accept this review?"),
            feature,
        )
        if approved is True:
            state.resolved = True
        return state


class ResolutionPhase(Phase):
    name = "resolution"

    async def execute(self, runner, feature, state):
        if state.resolved:
            print("Review accepted — no fixes needed.")
            return state

        # Human chooses fix strategy
        strategy = await runner.run(
            Choose(
                chooser=human,
                prompt="How would you like to address the review?",
                options=["Refactor", "Add tests", "Document and defer", "Reject review"],
            ),
            feature,
        )
        state.strategy = strategy
        print(f"\nChosen strategy: {strategy}")

        if strategy == "Reject review":
            state.resolved = True
            return state

        # Agent applies the fix
        fix_result = await runner.run(
            Ask(
                actor=reviewer,
                prompt=f"Apply fix strategy '{strategy}' to the submission.",
                context_keys=["review"],
            ),
            feature,
        )
        print(f"\n--- Fix Applied ---\n{fix_result}\n")

        # Human confirms
        confirmed = await runner.run(
            Gate(approver=human, prompt="Confirm the fix is satisfactory?"),
            feature,
        )
        state.resolved = confirmed is True
        return state


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class CodeReviewWorkflow(Workflow):
    name = "code-review"

    def build_phases(self):
        return [SubmissionPhase, ResolutionPhase]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    auto = "--auto" in sys.argv

    store = InMemoryStore()
    workspace = Workspace(id="main", path=Path.cwd(), branch="main")

    interaction_runtime = AutoApproveRuntime() if auto else TerminalInteractionRuntime()

    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": EchoAgentRuntime(),
            "terminal": interaction_runtime,
        },
        stores={"artifacts": store},
        workspaces={"main": workspace},
    )

    feature = Feature(
        id="cr-001",
        name="Code Review",
        slug="code-review",
        workflow_name="code-review",
        workspace_id="main",
    )

    state = ReviewState()
    result = await runner.execute_workflow(CodeReviewWorkflow(), feature, state)
    print(f"\n=== Done ===\nResolved: {result.resolved}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
