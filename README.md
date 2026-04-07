# iriai-compose

Runtime-agnostic, actor-neutral workflow orchestration for multi-agent systems.

## Overview

iriai-compose provides a structured way to orchestrate workflows involving multiple agents and human interactions. It separates **what** happens (tasks and phases) from **who** does it (actors) and **how** it runs (runtimes), making workflows portable across different agent backends and interaction methods.

**The core value proposition:** write a workflow once, run it with any combination of runtimes. The same workflow works with Claude, GPT, a terminal UI, Slack, or a custom integration — swap the runtime config, not the workflow code.

### Key Concepts

- **Ask** — The atomic task type. One prompt, one actor, one response. Every other task composes Asks.
- **Composite Tasks** — `Gate`, `Choose`, `Respond`, `Interview` — higher-level interaction patterns built from Asks
- **Actors** — Either AI agents (`AgentActor`) or humans (`InteractionActor`), each with a `resolver` that routes to a runtime
- **Runtimes** — Pluggable backends that implement a single `ask()` method. `AgentRuntime` for LLMs, `InteractionRuntime` for human UIs
- **Stores** — Named, pluggable persistence backends that implement `get()`/`put()`/`delete()`. The runner holds multiple named stores via `stores: dict[str, Store]`
- **ContextProvider** — Resolves `context_keys` on actors and tasks into prompt-ready strings by searching across registered stores
- **Phases** — Orchestration units that group tasks with control flow
- **Workflows** — Reusable templates composed of sequential phases
- **Features** — Concrete execution instances binding identity to a workflow and workspace

## Installation

```bash
pip install iriai-compose
```

For interactive terminal prompts:

```bash
pip install iriai-compose[terminal]
```

## Quick Start

```python
import asyncio
from pathlib import Path
from pydantic import BaseModel

from iriai_compose import (
    AgentActor, AgentRuntime, Ask, DefaultWorkflowRunner,
    Feature, Gate, InMemoryStore, InteractionActor, Phase,
    Role, Workflow, Workspace,
)
from iriai_compose.runtimes import TerminalInteractionRuntime


# --- Define your agent runtime ---
class MyAgentRuntime(AgentRuntime):
    name = "my-agent"

    async def ask(self, task, **kwargs):
        prompt = task.to_prompt()
        context = kwargs.get("context", "")
        if context:
            prompt = f"{context}\n\n## Task\n{prompt}"
        return await your_llm_call(prompt)


# --- Define actors ---
reviewer = AgentActor(
    name="reviewer",
    role=Role(
        name="code-reviewer",
        prompt="You are an experienced code reviewer.",
        tools=["Read", "Glob", "Grep"],
    ),
    resolver="my-agent",
)
human = InteractionActor(name="developer", resolver="terminal")


# --- Define a phase ---
class ReviewPhase(Phase):
    name = "review"

    async def execute(self, runner, feature, state):
        review = await runner.run(
            Ask(actor=reviewer, prompt="Review the code."),
            feature,
        )
        approved = await runner.run(
            Gate(approver=human, prompt="Accept this review?"),
            feature,
        )
        return state


# --- Define a workflow ---
class ReviewWorkflow(Workflow):
    name = "review"
    def build_phases(self):
        return [ReviewPhase]


# --- Run it ---
async def main():
    runner = DefaultWorkflowRunner(
        runtimes={
            "my-agent": MyAgentRuntime(),
            "terminal": TerminalInteractionRuntime(),
        },
        stores={"artifacts": InMemoryStore()},
    )
    feature = Feature(
        id="f-1", name="Review", slug="review",
        workflow_name="review", workspace_id="main",
    )
    await runner.execute_workflow(ReviewWorkflow(), feature, BaseModel())

asyncio.run(main())
```

See [`examples/`](examples/) for complete runnable examples.

## Architecture

```
Workflow (template)
  Phase (orchestration unit)
    Task (Ask, Interview, Gate, Choose, Respond)
      Ask (atomic — the only task that reaches the runtime)
        Actor (AgentActor or InteractionActor)
          Runtime (AgentRuntime or InteractionRuntime)

Runner
  stores: dict[str, Store]         — named persistence backends
  context_provider: ContextProvider — resolves context_keys → prompt text
  services: dict[str, Any]         — application-level services
```

The execution flow:

```
runner.run(task)                           # lifecycle hooks (on_start/on_done)
  task.execute(runner, feature)            # task logic
    runner.run(Ask(...))                   # composite tasks create Asks
      Ask.execute(runner, feature)
        runner.resolve(ask, feature)       # context injection + routing
          runtime.ask(task, **kwargs)      # the single runtime method
```

- **Leaf tasks** (Ask) pass themselves to the runtime via `runner.resolve()`
- **Composite tasks** (Gate, Choose, Respond, Interview) orchestrate Asks via `runner.run()` in their `execute()`
- The **runner** handles context resolution and routes to runtimes via `actor.resolver`
- Each **runtime** receives the full Ask task and decides how to use its fields

### Built-in Tasks

| Task | Type | Description |
|------|------|-------------|
| `Ask` | Atomic | One-shot prompt/response. The only task that reaches the runtime. |
| `Gate` | Composite | Approval checkpoint. Presents Approve/Reject/Give feedback, returns `True`/`False`/`str`. |
| `Choose` | Composite | Selection from options. Wraps an Ask with `Select` input. |
| `Respond` | Composite | Free-form input. Wraps a plain Ask. |
| `Interview` | Composite | Multi-turn conversation loop between a questioner and responder. |

### Ask Fields

Ask separates instruction from data:

| Field | Purpose |
|-------|---------|
| `prompt` | The instruction (what to do) |
| `input` | The data (what to work with) — flows from previous steps |
| `input_type` | Schema for the input — runtimes can use it for presentation |
| `output_type` | Schema for structured responses |
| `to_prompt()` | Combines prompt + input into a single string for agents |

### Built-in Runtimes

| Runtime | Type | Description |
|---------|------|-------------|
| `TerminalInteractionRuntime` | Interaction | Interactive terminal prompts via questionary |
| `AutoApproveRuntime` | Interaction | Auto-approves all interactions (useful for testing) |

### Stores

Stores are named, pluggable persistence backends. Compose defines the `Store` ABC — applications provide implementations (in-memory, Postgres, filesystem, etc.).

The runner holds multiple named stores:

```python
runner = DefaultWorkflowRunner(
    stores={
        "artifacts": PostgresStore(...),
        "events": EventStore(...),
    },
    ...
)
```

Phases read and write through named stores:

```python
await runner.stores["artifacts"].put("prd", prd_text, feature=feature)
existing = await runner.stores["artifacts"].get("design", feature=feature)
```

All store operations are scoped by `feature` — no cross-feature data leakage.

| Store | Description |
|-------|-------------|
| `InMemoryStore` | Dict-backed store for development and testing |

### Context Resolution

Agents receive background information via `context_keys` — declared on actors and tasks:

```python
architect = AgentActor(
    name="architect",
    role=architect_role,
    context_keys=["project", "prd"],  # always injected
)

# Task-level keys are merged with actor-level keys:
await runner.run(
    Ask(actor=architect, prompt="Design the system", context_keys=["design"]),
    feature,
)
# → architect sees: project + prd + design
```

The runner resolves context_keys via `ContextProvider`, which is auto-constructed from the runner's stores when not provided explicitly.

Keys can be **plain** (`"prd"`) to scan all stores, or **namespaced** (`"artifacts.prd"`) to target a specific store directly.

## Writing Custom Runtimes

Every runtime implements a single method: `ask(task, **kwargs)`.

**Agent runtime** — receives the Ask task and sends it to an LLM:

```python
from iriai_compose import AgentRuntime

class ClaudeRuntime(AgentRuntime):
    name = "claude"

    async def ask(self, task, **kwargs):
        prompt = task.to_prompt()
        context = kwargs.get("context", "")
        if context:
            prompt = f"{context}\n\n## Task\n{prompt}"

        role = task.actor.role
        output_type = task.output_type

        return await self.client.invoke(
            role=role, prompt=prompt, output_type=output_type,
            workspace=kwargs.get("workspace"),
            session_key=kwargs.get("session_key"),
        )
```

**Interaction runtime** — receives the Ask task and presents it to a human:

```python
from iriai_compose import InteractionRuntime
from iriai_compose.prompts import Select, Confirm

class SlackRuntime(InteractionRuntime):
    name = "slack"

    async def ask(self, task, **kwargs):
        if isinstance(task.input, Select):
            return await self.post_buttons(task.prompt, task.input.options)
        elif isinstance(task.input, Confirm):
            return await self.post_confirm(task.prompt)
        else:
            return await self.post_text_input(task.prompt)
```

The runtime decides its own UX. The framework passes the Ask with all its fields — `prompt`, `input`, `input_type`, `output_type`, `to_prompt()` — and the runtime uses whichever fields it needs.

## Writing Custom Stores

Every store implements three methods: `get()`, `put()`, `delete()` — all scoped by feature.

```python
from iriai_compose import Store

class PostgresStore(Store):
    def __init__(self, pool):
        self.pool = pool

    async def get(self, key, *, feature):
        row = await self.pool.fetchrow(
            "SELECT value FROM artifacts WHERE feature_id=$1 AND key=$2 "
            "ORDER BY id DESC LIMIT 1",
            feature.id, key,
        )
        return json.loads(row["value"]) if row else None

    async def put(self, key, value, *, feature):
        await self.pool.execute(
            "INSERT INTO artifacts (feature_id, key, value) VALUES ($1, $2, $3)",
            feature.id, key, json.dumps(value),
        )

    async def delete(self, key, *, feature):
        await self.pool.execute(
            "DELETE FROM artifacts WHERE feature_id=$1 AND key=$2",
            feature.id, key,
        )
```

Register stores by name on the runner. Use multiple stores for different concerns:

```python
runner = DefaultWorkflowRunner(
    runtimes={...},
    stores={
        "artifacts": PostgresStore(pool),
        "events": EventStore(pool),
    },
)
```

## Writing Custom Tasks

Custom tasks compose Asks (or other tasks) in their `execute()` method:

```python
from iriai_compose import Task, Ask, AgentActor

class Panel(Task):
    """Two agents give opinions, a third decides."""
    panelists: list[AgentActor]
    judge: AgentActor
    prompt: str

    async def execute(self, runner, feature, **kwargs):
        # Run panelists in parallel
        opinions = await runner.parallel(
            [Ask(actor=p, prompt=self.prompt) for p in self.panelists],
            feature,
        )

        # Judge decides
        opinion_text = "\n\n---\n\n".join(str(o) for o in opinions)
        return await runner.run(
            Ask(actor=self.judge,
                prompt=f"Given these opinions:\n\n{opinion_text}\n\nDecide."),
            feature,
        )
```

Custom tasks work with any runtime — they only create Asks, and the runner routes each Ask to the right runtime based on the actor's resolver.

## Best Practices

**Keep workflows runtime-agnostic.** Workflows should never reference specific runtimes. Actors have a `resolver` that routes to a runtime at configuration time, not definition time. This enables the same workflow to run with different runtimes (terminal for development, Slack for production).

**Separate instruction from data.** Use `prompt` for the instruction ("Rank these scholarships") and `input` for the data (the scholarship list). This enables the same task to work with different data and lets runtimes present the data appropriately.

**Use `output_type` for structured agent responses.** When you need structured data from an agent (not just a string), pass a Pydantic model as `output_type`. The agent runtime will parse the response into the model.

**Chain outputs to inputs.** The `output_type` of one Ask can flow directly into the `input` of the next:

```python
scholarships = await runner.run(
    Ask(actor=agent, prompt="Find scholarships", output_type=ScholarshipList),
    feature,
)
choice = await runner.run(
    Ask(actor=human, prompt="Pick one to apply for",
        input=scholarships, input_type=ScholarshipList),
    feature,
)
```

**Override `to_prompt()` for custom input formatting.** The default appends input as JSON. For domain-specific formatting, subclass Ask:

```python
class ScholarshipAsk(Ask):
    def to_prompt(self):
        criteria = self.input
        return (
            f"{self.prompt}\n\n"
            f"Minimum grant: ${criteria.min_amount}\n"
            f"Field: {criteria.field}"
        )
```

**Use lifecycle hooks for cross-cutting concerns.** `on_start` and `on_done` on Tasks, Phases, and Workflows are the right place for logging, metrics, and validation — not inside `execute()`.

**Use stores for cross-phase data.** Within a phase, pass data directly through state or task outputs. Across phases, use `runner.stores["artifacts"].put()` and `context_keys` for automatic injection into agent prompts.

## Example Use Cases

### Code Review Workflow

An agent reviews code, a human approves:

```python
class ReviewPhase(Phase):
    name = "review"

    async def execute(self, runner, feature, state):
        description = await runner.run(
            Respond(responder=human, prompt="Describe the code to review:"),
            feature,
        )
        review = await runner.run(
            Ask(actor=reviewer, prompt=f"Review:\n\n{description}"),
            feature,
        )
        approved = await runner.run(
            Gate(approver=human, prompt="Accept this review?"),
            feature,
        )
        state.resolved = approved is True
        return state
```

**Swap the runtime:** replace `TerminalInteractionRuntime` with `SlackInteractionRuntime` and the same workflow runs in Slack. Replace the agent runtime to use a different LLM.

### Multi-Agent Feature Planning

PM interviews a human, architect designs, validators review in parallel:

```python
class DiscoveryPhase(Phase):
    name = "discovery"

    async def execute(self, runner, feature, state):
        result = await runner.run(
            Interview(
                questioner=pm_agent,
                responder=human,
                initial_prompt="What problem does this feature solve?",
                done=lambda r: "DONE" in str(r).upper(),
            ),
            feature,
        )
        state.prd = str(result)
        await runner.stores["artifacts"].put("prd", state.prd, feature=feature)
        return state

class ValidationPhase(Phase):
    name = "validation"

    async def execute(self, runner, feature, state):
        results = await runner.parallel(
            [
                Ask(actor=security_validator, prompt=f"Security review:\n\n{state.design}"),
                Ask(actor=perf_validator, prompt=f"Performance review:\n\n{state.design}"),
            ],
            feature,
        )
        approved = await runner.run(
            Gate(approver=human, prompt="Final approval?"), feature
        )
        state.validated = approved is True
        return state
```

**Value:** the same workflow supports Claude + terminal during development, Claude + Slack in production, or GPT + a custom web UI — zero workflow changes.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Proprietary — Iriai
