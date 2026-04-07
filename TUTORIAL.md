# Building a Feature Planning Workflow

This tutorial walks through building a production-style, multi-phase workflow using iriai-compose. The workflow mirrors the patterns used in iriai-build's planning pipeline: structured interviews, gate approvals, parallel validation, artifact flow between phases, and context injection into agent prompts.

By the end you'll have a workflow where:

- A PM agent interviews a human to produce a PRD
- An architect agent designs a system based on the PRD
- The human approves or requests revisions
- Two validators run in parallel
- Artifacts flow between phases via stores and context_keys

## Prerequisites

```bash
pip install iriai-compose
```

You'll also need an agent runtime that connects to an LLM. The examples below use a placeholder `MyAgentRuntime` — substitute your own (Claude SDK, OpenAI, etc.).

## Step 1: Define the State Model

State is a Pydantic model that flows through all phases. Each phase reads what it needs and updates its fields.

```python
from pydantic import BaseModel, Field


class PlanningState(BaseModel):
    scope: str = ""
    prd: str = ""
    design: str = ""
    system_design: str = ""
    validated: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)
```

Keep state flat. Phases persist detailed artifacts to stores — state carries summaries and flags.

## Step 2: Define Roles and Actors

Roles define expertise. Actors bind a role to an identity and declare what context they need.

```python
from iriai_compose import AgentActor, InteractionActor, Role

# --- Roles ---

pm_role = Role(
    name="product-manager",
    prompt=(
        "You are a senior product manager. Ask clarifying questions to "
        "build a detailed PRD. Cover goals, user journeys, requirements, "
        "acceptance criteria, and edge cases. When the PRD is complete, "
        "respond with DONE."
    ),
    tools=["Read", "Glob"],
    model="claude-sonnet-4-6",
)

architect_role = Role(
    name="architect",
    prompt=(
        "You are a software architect. Produce clear, actionable technical "
        "designs. Reference the PRD for requirements. Include API contracts, "
        "data models, and component boundaries."
    ),
    tools=["Read", "Glob", "Grep"],
    model="claude-opus-4-6",
)

security_role = Role(
    name="security-reviewer",
    prompt="You review designs for security vulnerabilities and OWASP risks.",
    tools=["Read"],
)

perf_role = Role(
    name="performance-reviewer",
    prompt="You review designs for performance bottlenecks and scalability.",
    tools=["Read"],
)


# --- Actors ---
# context_keys declare which store artifacts are injected into the agent's prompt.

pm = AgentActor(name="pm", role=pm_role, context_keys=["project"])
architect = AgentActor(
    name="architect", role=architect_role, context_keys=["project", "prd"]
)
security_validator = AgentActor(
    name="security", role=security_role, context_keys=["design"]
)
perf_validator = AgentActor(
    name="performance", role=perf_role, context_keys=["design"]
)

human = InteractionActor(name="user", resolver="terminal")
```

**Key patterns:**

- `context_keys=["project", "prd"]` means the architect automatically receives the project description and PRD in its prompt — no manual threading needed.
- The `resolver` field on actors routes to a runtime by name. `"terminal"` maps to `TerminalInteractionRuntime`, `"agent"` to your LLM runtime.
- The same role can back multiple actors with different names and context. Two security reviewers with different context_keys see different artifacts.

## Step 3: Build Phases

### Phase 1: Scoping

Seed context and gather the feature description from the human.

```python
from iriai_compose import Ask, Interview, Phase, Respond


class ScopingPhase(Phase):
    name = "scoping"

    async def execute(self, runner, feature, state):
        # Seed project context — available to all agents via context_keys
        await runner.stores["artifacts"].put(
            "project",
            "Project: acme-platform — B2B SaaS for workflow automation.",
            feature=feature,
        )

        # Gather feature description from the human
        description = await runner.run(
            Respond(responder=human, prompt="Describe the feature you want to build:"),
            feature,
        )
        state.scope = description
        await runner.stores["artifacts"].put("scope", description, feature=feature)
        return state
```

### Phase 2: Discovery (Interview)

The PM agent interviews the human in a multi-turn conversation to produce a PRD.

```python
class DiscoveryPhase(Phase):
    name = "discovery"

    async def execute(self, runner, feature, state):
        # PM interviews human — loops until PM says DONE
        result = await runner.run(
            Interview(
                questioner=pm,
                responder=human,
                initial_prompt=(
                    f"Let's define this feature.\n\n"
                    f"Scope: {state.scope}\n\n"
                    f"I'll ask questions to build a complete PRD. "
                    f"What problem does this feature solve?"
                ),
                done=lambda r: isinstance(r, str) and "DONE" in r.upper(),
            ),
            feature,
        )

        state.prd = str(result)
        await runner.stores["artifacts"].put("prd", state.prd, feature=feature)
        return state
```

**How Interview works internally:**

1. The questioner (PM) sends the initial prompt — context_keys inject `"project"` automatically
2. The PM's response goes to the responder (human) as a plain prompt
3. The human's answer goes back to the PM with `continuation=True` (same conversation, no re-injected context)
4. Loop repeats until `done()` returns True

### Phase 3: Design with Gate Approval

The architect designs the system, then the human approves or requests revisions.

```python
from iriai_compose import Gate


class DesignPhase(Phase):
    name = "design"

    async def execute(self, runner, feature, state):
        max_attempts = 3

        for attempt in range(max_attempts):
            # Architect produces a design — receives project + prd via context_keys
            design = await runner.run(
                Ask(
                    actor=architect,
                    prompt="Produce a technical design based on the PRD.",
                ),
                feature,
            )
            state.design = str(design)
            await runner.stores["artifacts"].put(
                "design", state.design, feature=feature
            )

            # Human gates the design
            approved = await runner.run(
                Gate(approver=human, prompt="Approve this design?"),
                feature,
            )

            if approved is True:
                return state

            if isinstance(approved, str):
                # Feedback string — architect will see it on next attempt
                # via continuation of the same session
                print(f"Revision requested: {approved}")

        # Exhausted retries — proceed with last version
        return state
```

**How Gate works:**

- Presents three options: Approve, Reject, Give feedback
- Returns `True` (approved), `False` (rejected), or a feedback string
- The feedback string is the human's free-form input

**The gate-and-revise pattern** (Interview → Gate → revision loop) is common in production workflows. Build it as a helper if you use it often:

```python
async def gate_and_revise(runner, feature, actor, approver, prompt, max_revisions=3):
    """Interview until done, gate the result, revise if rejected."""
    for attempt in range(max_revisions):
        result = await runner.run(
            Ask(actor=actor, prompt=prompt), feature
        )
        approved = await runner.run(
            Gate(approver=approver, prompt="Approve?"), feature
        )
        if approved is True:
            return result
        prompt = f"Revise based on feedback:\n\n{approved}\n\nPrevious:\n\n{result}"
    return result  # last attempt
```

### Phase 4: Parallel Validation

Two validators run concurrently, then the human does final sign-off.

```python
class ValidationPhase(Phase):
    name = "validation"

    async def execute(self, runner, feature, state):
        # Two validators in parallel — each gets "design" via context_keys
        results = await runner.parallel(
            [
                Ask(
                    actor=security_validator,
                    prompt="Review this design for security risks.",
                ),
                Ask(
                    actor=perf_validator,
                    prompt="Review this design for performance concerns.",
                ),
            ],
            feature,
        )

        security_review, perf_review = results

        # Final human sign-off
        summary = (
            f"## Security Review\n{security_review}\n\n"
            f"## Performance Review\n{perf_review}"
        )
        approved = await runner.run(
            Gate(approver=human, prompt=f"Final approval?\n\n{summary}"),
            feature,
        )
        state.validated = approved is True
        return state
```

**Parallel constraints:** the same actor cannot appear in multiple parallel tasks (session collision). Use distinct actors — even if they share a role:

```python
# Same role, different actor names — safe for parallel
security_validator = AgentActor(name="security", role=security_role, ...)
perf_validator = AgentActor(name="performance", role=perf_role, ...)
```

## Step 4: Compose the Workflow

```python
from iriai_compose import Workflow


class PlanningWorkflow(Workflow):
    name = "planning"

    def build_phases(self):
        return [ScopingPhase, DiscoveryPhase, DesignPhase, ValidationPhase]
```

## Step 5: Wire the Runner and Execute

```python
import asyncio
from pathlib import Path

from iriai_compose import DefaultWorkflowRunner, Feature, InMemoryStore, Workspace
from iriai_compose.runtimes import TerminalInteractionRuntime


async def main():
    runner = DefaultWorkflowRunner(
        runtimes={
            "agent": MyAgentRuntime(),  # your LLM runtime
            "terminal": TerminalInteractionRuntime(),
        },
        stores={"artifacts": InMemoryStore()},
        workspaces={"main": Workspace(id="main", path=Path.cwd())},
    )

    feature = Feature(
        id="plan-001",
        name="User Authentication",
        slug="user-auth",
        workflow_name="planning",
        workspace_id="main",
    )

    state = PlanningState()
    result = await runner.execute_workflow(PlanningWorkflow(), feature, state)
    print(f"Validated: {result.validated}")


asyncio.run(main())
```

**What the runner does:**

- `stores={"artifacts": InMemoryStore()}` — named store for workflow artifacts. Swap with `PostgresStore(pool)` for production.
- `context_provider` auto-constructs from stores when not provided. It resolves `context_keys` by searching all registered stores.
- `runtimes` maps resolver keys to runtime instances. Actor resolvers (`"agent"`, `"terminal"`) look up from this dict.

## Context Flow Summary

Here's how data flows through this workflow:

```
ScopingPhase
  writes: artifacts.project, artifacts.scope

DiscoveryPhase
  pm reads: "project" (via context_keys)
  writes: artifacts.prd

DesignPhase
  architect reads: "project", "prd" (via context_keys)
  writes: artifacts.design

ValidationPhase
  security reads: "design" (via context_keys)
  performance reads: "design" (via context_keys)
```

Phases write artifacts to stores. Agents declare what they need via `context_keys`. The runner resolves keys from stores and injects the content into the agent's prompt. No manual data threading.

Keys can be **plain** (`"prd"`) to scan all stores, or **namespaced** (`"artifacts.prd"`) to target a specific store directly.

## Structured Outputs

For agents that should return structured data instead of free-form text, use `output_type`:

```python
from pydantic import BaseModel


class DesignDoc(BaseModel):
    summary: str
    components: list[str]
    api_contracts: list[str]
    data_models: list[str]
    risks: list[str]


design = await runner.run(
    Ask(
        actor=architect,
        prompt="Produce a technical design.",
        output_type=DesignDoc,
    ),
    feature,
)
# design is a DesignDoc instance — the runtime parses it
```

Your agent runtime is responsible for parsing the structured output. The framework passes `output_type` through to the runtime via `task.output_type`.

## Lifecycle Hooks

Tasks, phases, and workflows support `on_start` and `on_done` hooks:

```python
class DesignPhase(Phase):
    name = "design"

    async def on_start(self, runner, feature, state):
        # Check for existing approved artifact — skip if already done
        existing = await runner.stores["artifacts"].get("design", feature=feature)
        if existing:
            state.design = existing

    async def on_done(self, runner, feature, state):
        print(f"Design phase complete. Has design: {bool(state.design)}")

    async def execute(self, runner, feature, state):
        if state.design:
            return state  # skip — already done
        # ... normal design logic
```

Hooks fire in order: `workflow.on_start` → `phase.on_start` → `phase.execute` → `phase.on_done` → `workflow.on_done`.

## Swapping Runtimes

The same workflow runs with different runtimes — no code changes:

```python
# Development: terminal + echo agent
runner = DefaultWorkflowRunner(
    runtimes={
        "agent": EchoAgentRuntime(),
        "terminal": TerminalInteractionRuntime(),
    },
    stores={"artifacts": InMemoryStore()},
)

# Production: Slack + Claude
runner = DefaultWorkflowRunner(
    runtimes={
        "agent": ClaudeRuntime(api_key=...),
        "terminal": SlackInteractionRuntime(channel=...),
    },
    stores={"artifacts": PostgresStore(pool)},
)

# Testing: auto-approve everything
from iriai_compose.runtimes import AutoApproveRuntime

runner = DefaultWorkflowRunner(
    runtimes={
        "agent": MockAgentRuntime(response="DONE"),
        "terminal": AutoApproveRuntime(),
    },
    stores={"artifacts": InMemoryStore()},
)
```

## Summary

| Concept | What it does | Where it lives |
|---------|-------------|----------------|
| **State** | Carries data between phases | Pydantic model, returned from each phase |
| **Stores** | Persist artifacts across phases | `runner.stores["name"]` — named, pluggable |
| **context_keys** | Declare what data agents need | On `AgentActor` and `Ask` |
| **ContextProvider** | Resolves keys from stores into prompts | Auto-constructed from stores |
| **Runtimes** | Execute Asks against LLMs or humans | `runner.runtimes` — named, pluggable |
| **Phases** | Group tasks with control flow | Sequential within a workflow |
| **Lifecycle hooks** | Setup/teardown at any level | `on_start`/`on_done` on tasks, phases, workflows |
