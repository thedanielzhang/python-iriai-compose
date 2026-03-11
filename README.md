# iriai-compose

Runtime-agnostic, actor-neutral workflow orchestration for multi-agent systems.

## Overview

iriai-compose provides a structured way to orchestrate workflows involving multiple agents and human interactions. It separates **what** happens (tasks and phases) from **who** does it (actors) and **how** it runs (runtimes), making workflows portable across different agent backends and interaction methods.

### Key Concepts

- **Actors** — Either AI agents (`AgentActor`) or humans (`InteractionActor`), each with a defined role
- **Tasks** — Atomic interaction patterns:
  - `Ask` — One-shot prompt/response
  - `Interview` — Multi-turn conversation loop with a termination condition
  - `Gate` — Approval/rejection checkpoint
  - `Choose` — Selection from a list of options
  - `Respond` — Free-form open-ended input
- **Phases** — Orchestration units that group tasks with control flow
- **Workflows** — Reusable templates composed of sequential phases
- **Features** — Concrete execution instances binding identity to a workflow and workspace
- **Runtimes** — Pluggable backends for agent execution (`AgentRuntime`) and human interaction (`InteractionRuntime`)

## Installation

```bash
pip install iriai-compose
```

For the Claude Agent SDK runtime:

```bash
pip install iriai-compose[claude]
```

## Quick Start

```python
import asyncio
from pathlib import Path
from pydantic import BaseModel

from iriai_compose import (
    AgentActor, Ask, DefaultContextProvider, DefaultWorkflowRunner,
    Feature, Gate, InMemoryArtifactStore, InteractionActor,
    Phase, Role, Workflow, Workspace,
)
from iriai_compose.runtimes import TerminalInteractionRuntime

# Define actors
reviewer = AgentActor(
    name="reviewer",
    role=Role(
        name="code-reviewer",
        prompt="You are an experienced code reviewer.",
        tools=["Read", "Glob", "Grep"],
    ),
)
human = InteractionActor(name="developer", resolver="terminal")

# Define a phase
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

# Define a workflow
class ReviewWorkflow(Workflow):
    name = "review"
    def build_phases(self):
        return [ReviewPhase]

# Run it
async def main():
    artifacts = InMemoryArtifactStore()
    runner = DefaultWorkflowRunner(
        agent_runtime=your_agent_runtime,  # plug in your AgentRuntime
        interaction_runtimes={"terminal": TerminalInteractionRuntime()},
        artifacts=artifacts,
        context_provider=DefaultContextProvider(artifacts=artifacts),
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
  └── Phase (orchestration unit)
        └── Task (Ask, Interview, Gate, Choose, Respond)
              └── Actor (AgentActor or InteractionActor)
                    └── Runtime (AgentRuntime or InteractionRuntime)
```

The `WorkflowRunner` coordinates everything. Phases call `runner.run(task)`, tasks call `runner.resolve(actor)`, and the runner dispatches to the appropriate runtime based on actor type.

### Built-in Runtimes

| Runtime | Type | Description |
|---------|------|-------------|
| `ClaudeAgentRuntime` | Agent | Claude Agent SDK with session resumption and structured output |
| `TerminalInteractionRuntime` | Interaction | Interactive terminal prompts via questionary |
| `AutoApproveRuntime` | Interaction | Auto-approves all gates (useful for testing) |

### Storage

- **ArtifactStore** — Key-value store for intermediate results shared between phases
- **SessionStore** — Persists agent sessions for resumption across turns
- **ContextProvider** — Resolves context keys into prompt-ready strings

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Proprietary — Iriai
