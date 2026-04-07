"""Example B: Multi-actor feature planning workflow.

Three-phase workflow exercising Interview, parallel tasks,
multiple actors with different context_keys, and artifacts flowing
through context resolution.

Run interactively:   python examples/feature_planning.py
Run auto-approved:   python examples/feature_planning.py --auto
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel

from iriai_compose import (
    AgentActor,
    Ask,
    DefaultWorkflowRunner,
    Feature,
    Gate,
    InMemoryStore,
    InteractionActor,
    Interview,
    Phase,
    Role,
    Workflow,
    Workspace,
)
from iriai_compose.runtimes import AutoApproveRuntime, TerminalInteractionRuntime

from runtime import EchoAgentRuntime


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanningState(BaseModel):
    description: str = ""
    prd: str = ""
    design: str = ""
    validated: bool = False


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------

pm_role = Role(
    name="product-manager",
    prompt="You are a product manager. Ask clarifying questions to build a PRD.",
    tools=["Read"],
)

architect_role = Role(
    name="architect",
    prompt="You are a software architect. Produce clear, actionable designs.",
    tools=["Read", "Glob", "Grep"],
    model="claude-opus-4-6",
)

validator_role = Role(
    name="validator",
    prompt="You validate designs for feasibility and completeness.",
    tools=["Read"],
)

pm_agent = AgentActor(name="pm", role=pm_role, context_keys=["project"])
architect_agent = AgentActor(
    name="architect", role=architect_role, context_keys=["project", "prd"]
)
security_validator = AgentActor(name="security-validator", role=validator_role)
perf_validator = AgentActor(name="perf-validator", role=validator_role)

human = InteractionActor(name="user", resolver="terminal")


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

class DiscoveryPhase(Phase):
    """PM interviews the human to produce a PRD."""

    name = "discovery"

    async def execute(self, runner, feature, state):
        # Seed project context
        await runner.stores["artifacts"].put(
            "project",
            "Project: iriai-sdk — workflow orchestration library for AI agents.",
            feature=feature,
        )

        # Interview: PM asks questions, human answers, loop until PM says DONE
        result = await runner.run(
            Interview(
                questioner=pm_agent,
                responder=human,
                initial_prompt="Let's define the feature. What problem does it solve?",
                done=lambda r: isinstance(r, str) and "DONE" in r.upper(),
            ),
            feature,
        )
        state.prd = str(result)
        await runner.stores["artifacts"].put("prd", state.prd, feature=feature)
        print(f"\n--- PRD ---\n{state.prd}\n")
        return state


class DesignPhase(Phase):
    """Architect produces a design; human approves or requests revision."""

    name = "design"

    async def execute(self, runner, feature, state):
        for attempt in range(3):
            design = await runner.run(
                Ask(
                    actor=architect_agent,
                    prompt="Produce a technical design based on the PRD.",
                ),
                feature,
            )
            state.design = str(design)
            await runner.stores["artifacts"].put("design", state.design, feature=feature)
            print(f"\n--- Design (attempt {attempt + 1}) ---\n{state.design}\n")

            approved = await runner.run(
                Gate(approver=human, prompt="Approve this design?"),
                feature,
            )
            if approved is True:
                return state
            # Feedback string fed back as context for next attempt
            print(f"Revision requested: {approved}")

        print("Design phase exhausted retries — proceeding with last version.")
        return state


class ValidationPhase(Phase):
    """Two validators run in parallel; human does final sign-off."""

    name = "validation"

    async def execute(self, runner, feature, state):
        # Parallel validation
        results = await runner.parallel(
            [
                Ask(
                    actor=security_validator,
                    prompt=f"Security review of design:\n\n{state.design}",
                ),
                Ask(
                    actor=perf_validator,
                    prompt=f"Performance review of design:\n\n{state.design}",
                ),
            ],
            feature,
        )
        print("\n--- Validation Results ---")
        for label, res in zip(["Security", "Performance"], results):
            print(f"  {label}: {res}")

        # Final human approval
        approved = await runner.run(
            Gate(approver=human, prompt="Final approval — ship it?"),
            feature,
        )
        state.validated = approved is True
        return state


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class FeaturePlanningWorkflow(Workflow):
    name = "feature-planning"

    def build_phases(self):
        return [DiscoveryPhase, DesignPhase, ValidationPhase]


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
        id="fp-001",
        name="Feature Planning",
        slug="feature-planning",
        workflow_name="feature-planning",
        workspace_id="main",
    )

    state = PlanningState()
    result = await runner.execute_workflow(FeaturePlanningWorkflow(), feature, state)
    print(f"\n=== Done ===\nValidated: {result.validated}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
