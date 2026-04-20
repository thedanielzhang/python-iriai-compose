# iriai-compose: Unified Library Specification

> Historical note
> This spec references `Interview`, `Gate`, `Choose`, and `Respond` as design
> patterns, but they are no longer exported by the public `iriai-compose`
> library API. Current example implementations live under
> [`examples/_composite_tasks.py`](examples/_composite_tasks.py).

A Python library for composing multi-agent workflows. Runtime-agnostic core, ergonomic phase authoring, actor-neutral task definitions.

---

## Lineage

This spec synthesizes two prior documents:

- `SPEC.md` — concrete, ergonomic, Claude-native. Good phase readability, Interview primitive, Pending model. Too coupled to Claude SDK, bakes persistence into the library.
- `LIBRARY-SPEC.md` — abstract, runtime-agnostic, clean separation. Good adapter boundaries, WorkflowRunner mediator. Over-specified task I/O, lost Interview and Pending, verbose phase authoring.

This document takes the architecture of LIBRARY-SPEC and the ergonomics of SPEC.

---

## Positioning

iriai-compose occupies different territory from LangChain and LangGraph, despite all being in the agent orchestration space.

**iriai does not own the LLM call.** LangChain and LangGraph are opinionated about how you talk to models — prompt formatting, model selection, output parsing, tool execution. iriai pushes all of that to runtime adapters. The core library never touches a model. This makes it thinner and runtime-agnostic: swap Claude for Codex or a local model by changing the adapter, not the orchestration logic.

**Imperative scripts, not declared graphs.** LangGraph requires defining a graph topology upfront — nodes, edges, conditional routing. iriai's phases are `async def` functions with normal Python control flow (`if`, `while`, `for`, `try/except`). A phase reads like a narrative of what happens, not a wiring diagram. This trades static analyzability for readability and flexibility.

**Actors and interaction patterns are first-class.** LangChain has no concept of "a human and an agent having a multi-turn conversation." LangGraph can model it by building a graph with interrupt nodes, but there's no abstraction for the *pattern*. In iriai, `Interview(questioner=agent, responder=human)` is a primitive. The same `Interview` works with two agents, an agent and a human, or a human and an agent — the pattern is decoupled from who fills the slots.

**When to use what:**
- **LangChain** — single-agent tool-use pipelines where you want the framework to handle prompt/model/parsing
- **LangGraph** — complex agent state machines with checkpointing and production-grade durability
- **iriai** — multi-actor workflows where humans and agents collaborate through structured interaction patterns, with runtime-agnostic orchestration

---

## Design Principles

1. **Phases read like scripts.** A Phase should read top-to-bottom as a clear narrative of what happens. Minimize ceremony.
2. **Tasks define interaction patterns between actors.** A Task orchestrates how actors communicate — one-shot, multi-turn, approval, selection. Tasks are not constrained to one actor or one actor type.
3. **Actors are abstract.** An actor may be an AI agent, a human, or an automated policy. Tasks don't know or care. The runner resolves actors to runtimes.
4. **Actors carry their own state.** An `AgentActor` defines its baseline context (what artifacts it needs) and session identity (using the same actor object twice continues the same conversation). This eliminates repetitive wiring in phases.
5. **Phases own control flow.** Branching, loops, retries, and conditional logic live in Phases. Parallelism and child workflows are orchestration, not tasks.
6. **Runtime adapters, not runtime coupling.** The core library does not import Claude SDK, Codex SDK, or any specific agent framework. Adapters do.
7. **Storage is abstracted.** The library defines store interfaces. Applications choose backends.

---

## Core Hierarchy

```
WorkflowRunner      — coordinator       — runs tasks, resolves actors to runtimes
Workflow            — template          — reusable phase sequence
Phase               — orchestration     — tasks + control flow
Task                — interaction pattern — defines how actors communicate
Actor               — participant       — agent, human, or policy
Role                — expertise         — prompt, tools, model preference (for agent actors)
Workspace           — environment       — where agents execute (cwd, branch)
Feature             — instance          — binds workflow + workspace + identity
ArtifactStore       — persistence       — keyed document storage
ContextProvider     — resolver          — keys → prompt-ready context string
Pending             — suspension point  — what's waiting on external input
```

### Ownership

```
WorkflowRunner
 ├── AgentRuntime(s)         (pluggable, keyed by name)
 ├── InteractionRuntime(s)   (pluggable, keyed by name)
 ├── ArtifactStore           (pluggable)
 ├── SessionStore            (optional, pluggable)
 ├── ContextProvider         (pluggable)
 └── Feature (many, concurrent)
      ├── Workflow (one, shared template)
      │    └── Phase (sequential)
      │         └── Task (atomic)
      └── Workspace (one primary + N child workspaces)
```

### Layered Responsibilities

```
Phase           — what tasks to run and in what order     — calls runner.run(task)
Task            — how actors interact                     — calls runner.resolve(actor)
Runner          — routes actors to runtimes               — calls agent_runtime / interaction_runtime
Runtime         — actually executes                       — Claude SDK, Slack, terminal, etc.
```

Each layer has exactly one job.

---

## Actor

An actor is any entity that can receive a prompt and produce a response. Tasks are defined in terms of actors, not in terms of "agent" or "human."

```python
class Actor(BaseModel):
    name: str
```

### AgentActor

An actor backed by an AI agent runtime. Carries a Role, its own context, and session identity.

```python
class AgentActor(Actor):
    """Resolved by an AgentRuntime."""
    role: Role
    context_keys: list[str] = Field(default_factory=list)  # baseline context
    persistent: bool = True                                 # maintain session across resolves
```

**Session identity:** The runner derives the session key from `(actor.name, feature.id)`. Using the same `AgentActor` instance across multiple `runner.resolve()` calls continues the same conversation. The agent remembers prior turns.

```python
pm_agent = AgentActor(name="pm", role=pm, context_keys=["project"])

# First resolve — starts a session
await runner.resolve(pm_agent, "Analyze this feature", feature=feature)

# Second resolve — same actor, same session, agent remembers turn 1
await runner.resolve(pm_agent, "Now consider this feedback", feature=feature)
```

To create separate sessions for the same role, use different actor names:

```python
pm_planning = AgentActor(name="pm-planning", role=pm, context_keys=["project"])
pm_review = AgentActor(name="pm-review", role=pm, context_keys=["project", "prd"])
```

For truly stateless one-shot invocations, opt out:

```python
validator = AgentActor(name="validator", role=plan_compiler, persistent=False)
```

**Baseline context:** `context_keys` defines what artifacts this actor always needs to see. The `ContextProvider` resolves keys to content; missing keys (artifacts not yet created) resolve to nothing. As the workflow progresses and artifacts accumulate, the actor's view naturally fills in.

```python
# Architect always sees project, prd, design — even if prd/design don't exist yet
architect_agent = AgentActor(
    name="architect", role=architect,
    context_keys=["project", "prd", "design"],
)
```

Tasks can add task-specific context on top (see Task section). The runner merges and deduplicates.

### InteractionActor

An actor backed by an interaction runtime. Could be a human, another agent acting as reviewer, or an automated policy.

```python
class InteractionActor(Actor):
    """Resolved by an InteractionRuntime."""
    resolver: str  # "human.slack", "human.terminal", "auto", "agent.reviewer"
```

The `resolver` string is a routing key. The runner maps it to an `InteractionRuntime` instance. The task doesn't know what's behind it.

InteractionActors don't carry `context_keys` or session state — humans see the prompt directly, and each interaction is self-contained.

---

## Role

A Role defines expertise, perspective, and capabilities for an `AgentActor`. No interaction logic, no output format, no orchestration instructions.

```python
class Role(BaseModel):
    name: str                              # identity
    prompt: str                            # expertise/perspective (2-3 sentences)
    tools: list[str] = Field(default_factory=list)   # capabilities
    model: str | None = None               # model preference (runtime interprets)
    metadata: dict[str, Any] = Field(default_factory=dict)  # runtime-specific extras
```

`tools` is on Role because "this agent can read files and run commands" is a statement about capability, not about which SDK you're using. The runtime adapter maps tool names to its own tool system.

`model` is a preference, not a directive. The runtime adapter decides how to honor it.

`metadata` carries runtime-specific settings (e.g. `setting_sources` for Claude, sandbox config for Codex) without polluting the core model.

**Guidelines for Role prompts:**
- Describe expertise and perspective only
- No orchestration instructions — that's the Phase's job
- No output format instructions — that's the Task's `output_type` parameter

**Example:**
```python
pm = Role(
    name="pm",
    prompt="You are a Product Manager. You understand user needs, business "
           "value, and requirement clarity. You identify gaps and ambiguities "
           "in specifications.",
    tools=["Read", "Glob", "Grep"],
)

architect = Role(
    name="architect",
    prompt="You understand system design, dependency graphs, and technical "
           "trade-offs. You think in interfaces and data flow.",
    tools=["Read", "Glob", "Grep", "Bash"],
    model="claude-opus-4-6",
)
```

---

## Task

A task defines an interaction pattern between actors. Each task type declares the actor slots that make sense for its semantics.

### Base

```python
class Task(ABC):
    context_keys: list[str] = Field(default_factory=list)  # task-specific additions

    @abstractmethod
    async def execute(self, runner: "WorkflowRunner", feature: "Feature") -> Any:
        """Define the interaction pattern. Call runner.resolve() to talk to actors."""
        ...
```

Tasks do not execute against runtimes directly. They call `runner.resolve(actor, prompt)`, which routes to the correct runtime. The task owns the *pattern* — how many turns, who talks when, what terminates the loop. The runner owns the *dispatch* — which runtime handles each actor.

**Context merging:** When the runner resolves an `AgentActor`, it merges `actor.context_keys` (baseline) with `task.context_keys` (task-specific), deduplicating. Most tasks leave `context_keys` empty because the actor already carries what it needs. Use task-level keys only for one-off context needs:

```python
# Architect's baseline is ["project", "prd", "design"]
# This specific task also needs the threat model
Ask(actor=architect_agent, prompt="Review the threat model", context_keys=["threat-model"])
# Runner resolves: ["project", "prd", "design", "threat-model"]
```

### Built-in Task Types

```python
class Ask(Task):
    """One-shot: send prompt to one actor, get result."""

    actor: Actor
    prompt: str
    output_type: type[BaseModel] | None = None

    async def execute(self, runner, feature):
        return await runner.resolve(
            self.actor, self.prompt,
            feature=feature,
            context_keys=self.context_keys,
            output_type=self.output_type,
        )


class Interview(Task):
    """Multi-turn: questioner asks, responder answers, loop until
    the termination condition is met.

    Session continuity is automatic — the questioner's AgentActor
    (if persistent) maintains conversation state across turns.

    The caller must provide a `done` predicate that inspects each
    questioner response and returns True when the interview should
    end. This makes the termination contract explicit rather than
    relying on implicit field conventions."""

    questioner: Actor
    responder: Actor
    initial_prompt: str
    output_type: type[BaseModel] | None = None
    done: Callable[[Any], bool]  # termination predicate — receives questioner's response

    async def execute(self, runner, feature):
        response = await runner.resolve(
            self.questioner, self.initial_prompt,
            feature=feature,
            context_keys=self.context_keys,
        )

        while True:
            answer = await runner.resolve(
                self.responder, str(response),
                feature=feature,
            )
            result = await runner.resolve(
                self.questioner, f"Response: {answer}",
                feature=feature,
                output_type=self.output_type,
            )
            if self.done(result):
                return result
            response = str(result)


class Gate(Task):
    """Approval: one actor approves, rejects, or gives feedback."""

    approver: Actor
    prompt: str

    async def execute(self, runner, feature):
        return await runner.resolve(
            self.approver, self.prompt,
            feature=feature,
            kind="approve",
        )


class Choose(Task):
    """Selection: one actor picks from options."""

    chooser: Actor
    prompt: str
    options: list[str]

    async def execute(self, runner, feature):
        return await runner.resolve(
            self.chooser, self.prompt,
            feature=feature,
            kind="choose",
            options=self.options,
        )


class Respond(Task):
    """Free-form: one actor provides open-ended input."""

    responder: Actor
    prompt: str

    async def execute(self, runner, feature):
        return await runner.resolve(
            self.responder, self.prompt,
            feature=feature,
            kind="respond",
        )
```

### Actor Flexibility

Because actors are abstract, the same task type supports many configurations:

```python
# Agent interviews human (original use case)
# PM carries its own context and maintains session automatically
pm_agent = AgentActor(name="pm", role=pm, context_keys=["project"])

await runner.run(Interview(
    questioner=pm_agent,
    responder=InteractionActor(name="user", resolver="human.slack"),
    initial_prompt="What questions do you have about this feature?",
    output_type=PRD,
    done=lambda r: isinstance(r, PRD) and not r.questions,
), feature)

# Agent interviews agent (no human in the loop)
# Both agents maintain separate sessions, carry their own context
await runner.run(Interview(
    questioner=AgentActor(name="pm", role=pm, context_keys=["project"]),
    responder=AgentActor(name="architect", role=architect, context_keys=["project", "prd"]),
    initial_prompt="What technical constraints should the PRD account for?",
    output_type=PRD,
    done=lambda r: isinstance(r, PRD) and not r.questions,
), feature)

# Human interviews agent (human drives, agent answers)
await runner.run(Interview(
    questioner=InteractionActor(name="tech-lead", resolver="human.slack"),
    responder=AgentActor(name="architect", role=architect, context_keys=["project", "design"]),
    initial_prompt="Explain your architecture choices.",
    output_type=ArchReview,
    done=lambda r: isinstance(r, ArchReview),
), feature)

# Gate approved by auto-policy instead of human
await runner.run(Gate(
    approver=InteractionActor(name="auto", resolver="auto-approve"),
    prompt="Approve PRD?",
), feature)

# Gate approved by an AI reviewer (stateless — fresh each time)
await runner.run(Gate(
    approver=AgentActor(name="security-reviewer", role=security_reviewer,
                        context_keys=["project", "plan"], persistent=False),
    prompt="Does this plan have security concerns?",
), feature)
```

### Custom Task Types

New interaction patterns are defined by creating new Task subclasses with their own actor slots:

```python
class Debate(Task):
    """Two actors argue positions, a judge decides.
    Each actor carries its own context and session state."""
    side_a: Actor
    side_b: Actor
    judge: Actor
    topic: str
    rounds: int = 3

    async def execute(self, runner, feature):
        history = []
        for _ in range(self.rounds):
            arg_a = await runner.resolve(self.side_a, f"Argue for: {self.topic}\n{history}",
                                         feature=feature)
            arg_b = await runner.resolve(self.side_b, f"Argue against: {self.topic}\n{history}",
                                         feature=feature)
            history.extend([arg_a, arg_b])

        return await runner.resolve(
            self.judge,
            f"Judge this debate:\n{history}",
            feature=feature,
            output_type=Verdict,
        )


class PanelReview(Task):
    """Multiple reviewers evaluate, results aggregated."""
    reviewers: list[Actor]
    prompt: str
    policy: Literal["all", "majority"] = "all"

    async def execute(self, runner, feature):
        # Resolve all reviewers in parallel
        import asyncio
        results = await asyncio.gather(*[
            runner.resolve(r, self.prompt, feature=feature, kind="approve")
            for r in self.reviewers
        ])
        if self.policy == "all":
            return all(r is True for r in results)
        return sum(1 for r in results if r is True) > len(results) / 2
```

---

## Pending

A suspension point where the workflow is waiting on external input.

`Pending` is a core library concept. It represents the *what* — what the workflow needs — without prescribing the *how* — how it gets persisted, delivered, or resolved.

```python
class Pending(BaseModel):
    id: str                                # unique identifier
    feature_id: str                        # which feature
    phase_name: str                        # which phase
    kind: Literal["approve", "choose", "respond"]
    prompt: str                            # what the resolver sees
    evidence: Any | None = None            # context for the decision
    options: list[str] | None = None       # for "choose" kind
    created_at: datetime
    resolved: bool = False
    response: str | bool | None = None
```

**Pending is serializable.** The application decides whether and how to persist it (memory, disk, database). Multiple Pendings can exist simultaneously.

The `InteractionRuntime` is responsible for presenting the Pending to a resolver and returning the response. The core library creates the Pending and awaits resolution; it does not assume the transport.

Note: Pending is only created for `InteractionActor` resolutions. When an `AgentActor` is resolved, the call goes directly to the `AgentRuntime` — no suspension, no Pending.

---

## Phase

Orchestration unit. Groups tasks with control flow. This is where branching, loops, retries, parallel fan-out, and child workflows live.

```python
class Phase(ABC):
    name: str

    @abstractmethod
    async def execute(self, runner: "WorkflowRunner", feature: "Feature",
                      state: BaseModel) -> BaseModel:
        ...
```

Phases talk to the runner via `runner.run(task, feature)`. They don't call `runner.resolve()` directly — that's the task's job.

### Example Phase

```python
# Define actors once with their baseline context and session identity.
# Context fills in as artifacts are created — missing keys resolve to nothing.
pm_agent = AgentActor(name="pm", role=pm, context_keys=["project"])
designer_agent = AgentActor(name="designer", role=designer, context_keys=["project", "prd"])
architect_agent = AgentActor(name="architect", role=architect, context_keys=["project", "prd", "design"])
plan_compiler_agent = AgentActor(name="plan-compiler", role=plan_compiler,
                                  context_keys=["project", "prd", "design", "plan"],
                                  persistent=False)  # stateless — fresh each validation
human = InteractionActor(name="user", resolver="human.slack")


class Planning(Phase):
    name = "planning"

    async def execute(self, runner, feature, state):
        # Interview — PM asks questions, human answers
        # PM's session persists across turns automatically
        prd = await runner.run(Interview(
            questioner=pm_agent,
            responder=human,
            initial_prompt=f"Analyze this feature request. What questions do you have?\n\n{state.description}",
            output_type=PRD,
            done=lambda r: isinstance(r, PRD) and not r.questions,
        ), feature)
        await runner.artifacts.put("prd", prd, feature=feature)

        # Gate — human approves PRD
        approved = await runner.run(Gate(approver=human, prompt="Approve PRD?"), feature)
        if isinstance(approved, str):
            # Feedback — PM revises (same session, remembers the interview)
            prd = await runner.run(Ask(
                actor=pm_agent,
                prompt=f"Revise the PRD using this feedback:\n{approved}",
                output_type=PRD,
            ), feature)
            await runner.artifacts.put("prd", prd, feature=feature)
            await runner.run(Gate(approver=human, prompt="Approve revised PRD?"), feature)

        # One-shot — designer produces design
        # Designer's context includes ["project", "prd"] — prd now exists
        design = await runner.run(Ask(
            actor=designer_agent,
            prompt="Produce design decisions for this feature.",
            output_type=DesignDecisions,
        ), feature)
        await runner.artifacts.put("design", design, feature=feature)

        # Choice — if designer proposes alternatives, human picks
        if design.alternatives:
            choice = await runner.run(Choose(
                chooser=human,
                prompt="Which approach?",
                options=design.alternatives,
            ), feature)
            design = await runner.run(Ask(
                actor=designer_agent,
                prompt=f"Finalize design with approach: {choice}",
                output_type=DesignDecisions,
            ), feature)
            await runner.artifacts.put("design", design, feature=feature)

        await runner.run(Gate(approver=human, prompt="Approve design?"), feature)

        # Interview — architect asks human technical questions
        # Architect's context includes ["project", "prd", "design"] — all exist now
        plan = await runner.run(Interview(
            questioner=architect_agent,
            responder=human,
            initial_prompt="What technical questions do you have about this feature?",
            output_type=Plan,
            done=lambda r: isinstance(r, Plan) and not r.questions,
        ), feature)

        # Loop — validate until plan compiles
        # Plan compiler is persistent=False — fresh context each attempt
        while True:
            verdict = await runner.run(Ask(
                actor=plan_compiler_agent,
                prompt="Validate this plan for completeness and feasibility.",
                output_type=Verdict,
            ), feature)
            if verdict.passed:
                break
            plan = await runner.run(Ask(
                actor=architect_agent,
                prompt=f"Fix these issues:\n{verdict.issues}",
                output_type=Plan,
            ), feature)
            await runner.artifacts.put("plan", plan, feature=feature)

        await runner.artifacts.put("plan", plan, feature=feature)
        await runner.run(Gate(approver=human, prompt="Approve plan for implementation?"), feature)

        state.plan = plan
        return state
```

---

## Workflow

A reusable template. Sequence of Phase types. Does not own execution state.

```python
class Workflow(ABC):
    name: str

    @abstractmethod
    def build_phases(self) -> list[type[Phase]]:
        """Return the ordered phase types for this workflow."""
        ...
```

`build_phases()` allows both static and dynamic phase lists.

**Example:**
```python
class FeaturePipeline(Workflow):
    name = "feature"

    def build_phases(self):
        return [Scoping, Planning, Implementation]


class HotfixPipeline(Workflow):
    name = "hotfix"

    def build_phases(self):
        return [HotfixPlanning, HotfixImplementation]
```

---

## Workspace

A physical environment where agents execute. Not a state store.

```python
class Workspace(BaseModel):
    id: str
    path: Path                             # the directory agents work in (cwd)
    branch: str | None = None              # git branch, if applicable
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Workspaces may be:
- A shared repo root
- A git worktree for isolation
- A temp directory
- A team-specific branch workspace

The runner provides workspace lookup:

```python
ws = runner.get_workspace("team-1")
```

How workspaces are created and managed (git worktree setup, cleanup, merging) is application-level. The library provides the concept and the lookup; the application provides the lifecycle.

---

## Feature

A concrete execution instance. Binds identity to a workflow and workspace.

```python
class Feature(BaseModel):
    id: str
    name: str
    slug: str
    workflow_name: str
    workspace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

The application may extend this (add `state`, `created_at`, etc.), but the core library only needs the binding.

---

## WorkflowRunner

The coordinator. Phases call `runner.run(task)`. Tasks call `runner.resolve(actor)`. The runner dispatches to runtimes.

```python
class WorkflowRunner(ABC):
    artifacts: ArtifactStore
    sessions: SessionStore | None
    context_provider: ContextProvider

    # --- Task execution (called by phases) ---

    async def run(self, task: Task, feature: Feature, *, phase_name: str = "") -> Any:
        """Execute a task. The task defines the interaction pattern.
        phase_name is threaded through to Pending creation."""
        self._current_phase = phase_name
        return await task.execute(self, feature)

    # --- Actor resolution (called by tasks) ---

    async def resolve(
        self,
        actor: Actor,
        prompt: str,
        *,
        feature: Feature,
        context_keys: list[str] | None = None,
        output_type: type[BaseModel] | None = None,
        kind: Literal["approve", "choose", "respond"] | None = None,
        options: list[str] | None = None,
    ) -> Any:
        """Route an actor to the correct runtime and return the response.

        For AgentActors: merges actor.context_keys + task context_keys,
        resolves context, manages session continuity automatically.

        For InteractionActors: creates a Pending and dispatches to
        the interaction runtime. Uses the current phase name from
        the enclosing run() call."""
        ...

    # --- Orchestration (called by phases) ---

    async def parallel(
        self,
        tasks: list[Task],
        feature: Feature,
        *,
        fail_fast: bool = True,
    ) -> list[Any]:
        """Run tasks concurrently.

        fail_fast=True (default): first exception cancels remaining tasks.
        fail_fast=False: all tasks run to completion; exceptions are collected
        and raised as an ExceptionGroup."""
        ...

    async def execute_child(
        self,
        workflow: Workflow,
        feature: Feature,
        state: BaseModel,
        *,
        workspace_id: str | None = None,
    ) -> BaseModel:
        """Execute a child workflow. If workspace_id is provided, the child
        workflow's feature is rebound to that workspace."""
        ...

    async def execute_workflow(
        self,
        workflow: Workflow,
        feature: Feature,
        state: BaseModel,
    ) -> BaseModel:
        """Execute a workflow's phases in sequence."""
        ...

    # --- Environment ---

    def get_workspace(self, workspace_id: str | None) -> Workspace | None:
        ...
```

### Default Implementation

```python
class DefaultWorkflowRunner(WorkflowRunner):
    def __init__(
        self,
        *,
        agent_runtime: AgentRuntime,
        interaction_runtimes: dict[str, InteractionRuntime],
        artifacts: ArtifactStore,
        sessions: SessionStore | None = None,
        context_provider: ContextProvider,
        workspaces: dict[str, Workspace] | None = None,
    ):
        self.agent_runtime = agent_runtime
        self.interaction_runtimes = interaction_runtimes
        self.artifacts = artifacts
        self.sessions = sessions
        self.context_provider = context_provider
        self._workspaces = workspaces or {}
        self._current_phase = ""

    def _resolve_interaction_runtime(self, resolver: str) -> InteractionRuntime:
        """Route a resolver key to an InteractionRuntime.
        Tries exact match first, then prefix match (e.g. "human.slack" -> "human")."""
        if resolver in self.interaction_runtimes:
            return self.interaction_runtimes[resolver]
        prefix = resolver.split(".")[0]
        if prefix in self.interaction_runtimes:
            return self.interaction_runtimes[prefix]
        raise KeyError(f"No InteractionRuntime registered for resolver '{resolver}'")

    async def resolve(self, actor, prompt, *, feature, context_keys=None,
                      output_type=None, kind=None, options=None):

        if isinstance(actor, AgentActor):
            # Merge context: actor baseline + task-specific, deduplicated
            all_keys = list(dict.fromkeys(actor.context_keys + (context_keys or [])))
            context_str = ""
            if all_keys:
                context_str = await self.context_provider.resolve(
                    all_keys, feature=feature
                )
            full_prompt = f"{context_str}\n\n## Task\n{prompt}" if context_str else prompt

            # Session key derived from actor identity + feature
            session_key = f"{actor.name}:{feature.id}" if actor.persistent else None

            # Dispatch to agent runtime
            workspace = self.get_workspace(feature.workspace_id)
            return await self.agent_runtime.invoke(
                role=actor.role,
                prompt=full_prompt,
                output_type=output_type,
                workspace=workspace,
                session_key=session_key,
            )

        elif isinstance(actor, InteractionActor):
            # Create Pending and dispatch to the correct interaction runtime
            pending = Pending(
                id=str(uuid4()),
                feature_id=feature.id,
                phase_name=self._current_phase,
                kind=kind or "respond",
                prompt=prompt,
                options=options,
                created_at=datetime.now(),
            )
            runtime = self._resolve_interaction_runtime(actor.resolver)
            return await runtime.resolve(pending)

        raise TypeError(f"Unknown actor type: {type(actor).__name__}")

    async def parallel(self, tasks, feature, *, fail_fast=True):
        if fail_fast:
            return await asyncio.gather(*[task.execute(self, feature) for task in tasks])
        else:
            results = await asyncio.gather(
                *[task.execute(self, feature) for task in tasks],
                return_exceptions=True,
            )
            exceptions = [r for r in results if isinstance(r, BaseException)]
            if exceptions:
                raise ExceptionGroup("parallel task failures", exceptions)
            return results

    async def execute_workflow(self, workflow, feature, state):
        for phase_cls in workflow.build_phases():
            phase = phase_cls()
            self._current_phase = phase.name
            state = await phase.execute(self, feature, state)
        return state

    def get_workspace(self, workspace_id):
        if workspace_id is None:
            return None
        return self._workspaces.get(workspace_id)
```

### Multiple Agent Runtimes

For applications that use multiple agent runtimes (Claude + Codex, Claude + local model):

```python
class MultiRuntimeWorkflowRunner(DefaultWorkflowRunner):
    def __init__(self, *, agent_runtimes: dict[str, AgentRuntime], **kwargs):
        self._agent_runtimes = agent_runtimes
        super().__init__(agent_runtime=next(iter(agent_runtimes.values())), **kwargs)

    async def resolve(self, actor, prompt, **kwargs):
        if isinstance(actor, AgentActor):
            runtime_name = actor.role.metadata.get("runtime", "default")
            runtime = self._agent_runtimes.get(runtime_name, self.agent_runtime)
            # Use this runtime for the invocation
            ...
        return await super().resolve(actor, prompt, **kwargs)
```

Multiple interaction runtimes are handled by the default runner via the `interaction_runtimes` dict, keyed by resolver prefix. See `_resolve_interaction_runtime()` above.

---

## Runtime Adapter Layer

The core library sits above runtime adapters. This is how it stays reusable across Claude SDK, Codex SDK, and future integrations.

### AgentRuntime

Executes agent invocations. Called by the runner when resolving an `AgentActor`.

```python
class AgentRuntime(ABC):
    name: str

    @abstractmethod
    async def invoke(
        self,
        role: Role,
        prompt: str,
        *,
        output_type: type[BaseModel] | None = None,
        workspace: Workspace | None = None,
        session_key: str | None = None,
    ) -> str | BaseModel:
        ...
```

**Adapter contract — implementations must:**
- Return a `BaseModel` instance when `output_type` is provided, `str` otherwise. If the response cannot be parsed into `output_type`, raise rather than returning a raw string.
- Handle `session_key` for conversation continuity. If the runtime supports sessions, use the key to resume. If it doesn't, ignore it — but never raise on a non-None `session_key`.
- Map `role.tools` to the runtime's own tool system. Tool names are abstract strings (e.g. `"Read"`, `"Bash"`); the adapter decides what they mean.
- Use `workspace.path` as the working directory if provided.
- Treat `role.model` as a preference, not a requirement. The adapter may fall back to a default if the requested model isn't available.
- Use `role.prompt` as the system prompt / agent identity.
- Read `role.metadata` for runtime-specific configuration (e.g. `setting_sources`, sandbox config). Unknown keys should be ignored.

**Example: Claude adapter**

```python
class ClaudeAgentRuntime(AgentRuntime):
    name = "claude"

    def __init__(self, session_store: SessionStore | None = None):
        self.session_store = session_store

    async def invoke(self, role, prompt, *, output_type=None,
                     workspace=None, session_key=None):
        options = ClaudeAgentOptions(
            system_prompt=role.prompt,
            allowed_tools=role.tools,
            model=role.model or "claude-sonnet-4-6",
            cwd=workspace.path if workspace else None,
        )
        if "setting_sources" in role.metadata:
            options.setting_sources = role.metadata["setting_sources"]

        # Session resumption
        if session_key and self.session_store:
            session = await self.session_store.load(session_key)
            if session and session.session_id:
                options.resume = session.session_id

        if output_type:
            options.output_format = output_type.model_json_schema()

        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                if session_key and self.session_store:
                    await self.session_store.save(AgentSession(
                        session_key=session_key,
                        session_id=msg.session_id,
                    ))
            if isinstance(msg, ResultMessage):
                if output_type:
                    return output_type.model_validate_json(msg.result)
                return msg.result
```

### InteractionRuntime

Resolves interaction requests. Called by the runner when resolving an `InteractionActor`.

Does not assume a human. The resolver may be a person, an agent, or an automated policy.

```python
class InteractionRuntime(ABC):
    name: str

    @abstractmethod
    async def resolve(self, pending: Pending) -> str | bool:
        """Present the pending to a resolver and return the response.
        May block for seconds (auto-approver) or days (human via Slack)."""
        ...
```

**Adapter contract — implementations must:**
- Handle all three `Pending.kind` values: `"approve"`, `"choose"`, `"respond"`.
- Return `bool` for `"approve"` (True = approved, False = rejected), or `str` for feedback.
- Return `str` for `"choose"` — one of the values from `pending.options`.
- Return `str` for `"respond"` — free-form input.
- May block indefinitely. The runner awaits the result; there is no built-in timeout. Implementations that need timeouts (e.g. SLA-based escalation) should handle them internally.
- Should use `pending.phase_name` and `pending.feature_id` for display/routing context.

**Example implementations:**

```python
class TerminalInteractionRuntime(InteractionRuntime):
    name = "terminal"

    async def resolve(self, pending):
        if pending.kind == "approve":
            response = input(f"\n{pending.prompt}\n[y/n/feedback]: ")
            if response.lower() == "y":
                return True
            elif response.lower() == "n":
                return False
            return response
        elif pending.kind == "choose":
            for i, opt in enumerate(pending.options):
                print(f"  {i+1}. {opt}")
            idx = int(input("Choice: ")) - 1
            return pending.options[idx]
        elif pending.kind == "respond":
            return input(f"\n{pending.prompt}\n> ")


class AutoApproveRuntime(InteractionRuntime):
    name = "auto"

    async def resolve(self, pending):
        if pending.kind == "approve":
            return True
        if pending.kind == "choose":
            return pending.options[0]
        return "auto-approved"
```

---

## Storage

The library defines store interfaces. Applications choose backends (memory, filesystem, SQL, etc.).

### ArtifactStore

Persists workflow outputs and reusable documents.

```python
class ArtifactStore(ABC):
    @abstractmethod
    async def get(self, key: str, *, feature: Feature) -> Any | None:
        ...

    @abstractmethod
    async def put(self, key: str, value: Any, *, feature: Feature) -> None:
        ...
```

Feature-scoped by default. The store implementation decides namespacing.

### SessionStore

Persists agent session data for continuity across invocations.

```python
class AgentSession(BaseModel):
    session_key: str
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStore(ABC):
    @abstractmethod
    async def load(self, session_key: str) -> AgentSession | None:
        ...

    @abstractmethod
    async def save(self, session: AgentSession) -> None:
        ...
```

Optional. If not provided, agents start fresh every invocation.

### ContextProvider

Resolves context keys to a prompt-ready string for agent invocations.

```python
class ContextProvider(ABC):
    @abstractmethod
    async def resolve(self, keys: list[str], *, feature: Feature) -> str:
        ...
```

Intentionally separate from artifact persistence. Context may come from artifacts, static files, computed functions, or external APIs.

**Example:**
```python
class DefaultContextProvider(ContextProvider):
    def __init__(self, artifacts: ArtifactStore, static_files: dict[str, Path]):
        self.artifacts = artifacts
        self.static_files = static_files

    async def resolve(self, keys, *, feature):
        sections = []
        for key in keys:
            if key in self.static_files:
                content = self.static_files[key].read_text()
            else:
                content = await self.artifacts.get(key, feature=feature)
            if content:
                sections.append(f"## {key}\n\n{content}")
        return "\n\n---\n\n".join(sections)
```

---

## Exceptions

The library defines a small exception hierarchy for its own failure modes. Runtime-specific errors (API failures, rate limits) are not wrapped — they propagate from the adapter and are accessible via `__cause__`.

```python
class IriaiError(Exception):
    """Base exception for all iriai library errors."""
    pass


class ResolutionError(IriaiError):
    """Actor could not be routed to a runtime.

    Raised when an InteractionActor's resolver key doesn't match any
    registered InteractionRuntime, or when an unknown Actor subclass
    is passed to runner.resolve()."""
    pass


class TaskExecutionError(IriaiError):
    """A task failed during execution.

    Wraps the underlying exception with context about which task,
    actors, phase, and feature were involved. The original exception
    is available via __cause__."""

    def __init__(self, *, task: "Task", feature: "Feature", phase_name: str):
        self.task = task
        self.feature = feature
        self.phase_name = phase_name
        actor_names = self._extract_actor_names(task)
        super().__init__(
            f"Task {type(task).__name__} failed in phase '{phase_name}' "
            f"for feature '{feature.id}' (actors: {actor_names})"
        )

    @staticmethod
    def _extract_actor_names(task):
        names = []
        for field_name in ['actor', 'questioner', 'responder', 'approver', 'chooser']:
            actor = getattr(task, field_name, None)
            if actor:
                names.append(actor.name)
        return ", ".join(names) or "unknown"
```

The runner's `run()` method wraps task failures automatically:

```python
async def run(self, task, feature, *, phase_name=""):
    self._current_phase = phase_name
    try:
        return await task.execute(self, feature)
    except IriaiError:
        raise  # don't double-wrap
    except Exception as e:
        raise TaskExecutionError(
            task=task,
            feature=feature,
            phase_name=self._current_phase,
        ) from e
```

Phase authors can inspect the cause for retry decisions:

```python
try:
    result = await runner.run(task, feature)
except TaskExecutionError as e:
    if isinstance(e.__cause__, RateLimitError):
        await asyncio.sleep(60)
        result = await runner.run(task, feature)  # retry
    else:
        raise
```

---

## In-Memory Storage Defaults

The core library ships in-memory implementations of all storage interfaces. These are suitable for development, testing, and short-lived workflows. Production deployments should use persistent backends (filesystem, database, etc.) provided by the application.

```python
class InMemoryArtifactStore(ArtifactStore):
    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}  # {feature_id: {key: value}}

    async def get(self, key, *, feature):
        return self._store.get(feature.id, {}).get(key)

    async def put(self, key, value, *, feature):
        self._store.setdefault(feature.id, {})[key] = value


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._sessions: dict[str, AgentSession] = {}

    async def load(self, session_key):
        return self._sessions.get(session_key)

    async def save(self, session):
        self._sessions[session.session_key] = session
```

`DefaultContextProvider` (shown in the Storage section above) is also shipped in the core library, backed by an `ArtifactStore` and optional static files.

---

## Implementation Requirements

- **Python**: 3.11+ (required for `ExceptionGroup`, `TaskGroup`)
- **Dependencies**: `pydantic>=2.0` (only hard dependency in core)
- **Runtime adapters** (e.g. Claude) require their own SDK installed separately

---

## Package Structure

```
iriai_compose/
  __init__.py            # re-exports public API
  actors.py              # Actor, AgentActor, InteractionActor, Role
  tasks.py               # Task, Ask, Interview, Gate, Choose, Respond
  workflow.py            # Phase, Workflow, Feature, Workspace
  runner.py              # WorkflowRunner, DefaultWorkflowRunner
  pending.py             # Pending
  storage.py             # ArtifactStore, SessionStore, ContextProvider (abstract + in-memory)
  exceptions.py          # IriaiError, ResolutionError, TaskExecutionError
  runtimes/
    __init__.py           # re-exports TerminalInteractionRuntime, AutoApproveRuntime
    claude.py             # ClaudeAgentRuntime (deferred import of Claude SDK)
```

Adapters with external dependencies (`claude.py`) use deferred imports — the module is importable, but instantiation raises a clear error if the SDK is not installed.

---

## What Is Core vs Application

### Core Library Defines

- `Actor`, `AgentActor`, `InteractionActor`
- `Role`
- `Task` (and built-in tasks: `Ask`, `Interview`, `Gate`, `Choose`, `Respond`)
- `Phase`, `Workflow`
- `Pending`
- `WorkflowRunner` (abstract + default implementation)
- `AgentRuntime`, `InteractionRuntime` (abstract)
- `ArtifactStore`, `SessionStore`, `ContextProvider` (abstract + in-memory defaults)
- `InMemoryArtifactStore`, `InMemorySessionStore`, `DefaultContextProvider`
- `TerminalInteractionRuntime`, `AutoApproveRuntime`
- `IriaiError`, `ResolutionError`, `TaskExecutionError`
- `Workspace`, `Feature`

### Application Layer Decides

- Persistence strategy (filesystem, database — in-memory defaults are provided)
- Crash recovery and state management
- Retry policies
- Workspace lifecycle (git worktree creation, cleanup, merging)
- Interaction delivery (Slack, web UI — terminal and auto-approve are provided)
- Agent runtime selection (Claude, Codex, local models)
- Scheduling and long-running process management
- Eventing and observability

---

## Full Pipeline Example

```python
from iriai_compose import (
    Role, AgentActor, InteractionActor,
    Workflow, Phase, Ask, Interview, Gate, Choose,
    DefaultWorkflowRunner, Feature, Workspace,
)

# --- Roles (expertise definitions, reusable) ---

pm = Role(
    name="pm",
    prompt="You understand user needs, business value, and requirement clarity. "
           "You identify gaps and ambiguities in specifications.",
    tools=["Read", "Glob", "Grep"],
)

designer = Role(
    name="designer",
    prompt="You understand user flows, component design, and interaction patterns.",
    tools=["Read", "Glob", "Grep"],
)

architect = Role(
    name="architect",
    prompt="You understand system design, dependency graphs, and technical trade-offs.",
    tools=["Read", "Glob", "Grep", "Bash"],
)

backend = Role(
    name="backend",
    prompt="You know FastAPI, SQLAlchemy, and async Python. You write production code.",
    tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
)

reviewer = Role(
    name="reviewer",
    prompt="You evaluate correctness, security, and adherence to spec.",
    tools=["Read", "Bash", "Glob", "Grep"],
)

plan_compiler = Role(
    name="plan-compiler",
    prompt="You validate implementation plans for completeness, consistency, "
           "and feasibility. You catch gaps and contradictions.",
    tools=["Read", "Glob", "Grep"],
)


# --- Actors (identity + context + session) ---

pm_agent = AgentActor(name="pm", role=pm,
                       context_keys=["project"])
designer_agent = AgentActor(name="designer", role=designer,
                             context_keys=["project", "prd"])
architect_agent = AgentActor(name="architect", role=architect,
                              context_keys=["project", "prd", "design"])
backend_agent = AgentActor(name="backend", role=backend,
                            context_keys=["project", "plan"])
reviewer_agent = AgentActor(name="reviewer", role=reviewer,
                             context_keys=["project", "plan"],
                             persistent=False)  # fresh review each time
plan_compiler_agent = AgentActor(name="plan-compiler", role=plan_compiler,
                                  context_keys=["project", "prd", "design", "plan"],
                                  persistent=False)  # stateless validation
human = InteractionActor(name="user", resolver="human.slack")


# --- Phases ---

class Planning(Phase):
    name = "planning"

    async def execute(self, runner, feature, state):
        # PM interviews human — session persists across turns
        prd = await runner.run(Interview(
            questioner=pm_agent,
            responder=human,
            initial_prompt=f"Analyze this feature request. What questions do you have?\n\n{state.description}",
            output_type=PRD,
            done=lambda r: isinstance(r, PRD) and not r.questions,
        ), feature)
        await runner.artifacts.put("prd", prd, feature=feature)
        await runner.run(Gate(approver=human, prompt="Approve PRD?"), feature)

        # Designer produces design — context auto-includes project + prd
        design = await runner.run(Ask(
            actor=designer_agent,
            prompt="Produce design decisions for this feature.",
            output_type=DesignDecisions,
        ), feature)
        await runner.artifacts.put("design", design, feature=feature)

        if design.alternatives:
            choice = await runner.run(Choose(
                chooser=human,
                prompt="Which approach?",
                options=design.alternatives,
            ), feature)
            design = await runner.run(Ask(
                actor=designer_agent,
                prompt=f"Finalize design with approach: {choice}",
                output_type=DesignDecisions,
            ), feature)
            await runner.artifacts.put("design", design, feature=feature)

        await runner.run(Gate(approver=human, prompt="Approve design?"), feature)

        # Architect interviews human — context auto-includes project + prd + design
        plan = await runner.run(Interview(
            questioner=architect_agent,
            responder=human,
            initial_prompt="What technical questions do you have about this feature?",
            output_type=Plan,
            done=lambda r: isinstance(r, Plan) and not r.questions,
        ), feature)

        # Validate until plan compiles
        while True:
            verdict = await runner.run(Ask(
                actor=plan_compiler_agent,
                prompt="Validate this plan for completeness and feasibility.",
                output_type=Verdict,
            ), feature)
            if verdict.passed:
                break
            plan = await runner.run(Ask(
                actor=architect_agent,
                prompt=f"Fix these issues:\n{verdict.issues}",
                output_type=Plan,
            ), feature)
            await runner.artifacts.put("plan", plan, feature=feature)

        await runner.artifacts.put("plan", plan, feature=feature)
        await runner.run(
            Gate(approver=human, prompt="Approve plan for implementation?"),
            feature,
        )

        state.plan = plan
        return state


class Implementation(Phase):
    name = "implementation"

    async def execute(self, runner, feature, state):
        for task_spec in state.plan.tasks:
            # Parallel implementation — backend_agent context auto-includes project + plan
            impl_tasks = [
                Ask(actor=backend_agent, prompt=f"Implement: {subtask.description}")
                for subtask in task_spec.subtasks
            ]
            results = await runner.parallel(impl_tasks, feature)

            # Review — reviewer context auto-includes project + plan
            review = await runner.run(Ask(
                actor=reviewer_agent,
                prompt="Review this implementation against the plan.",
            ), feature)

            await runner.run(
                Gate(approver=human, prompt=f"Approve {task_spec.name}?"),
                feature,
            )

        state.completed = True
        return state


# --- Workflow ---

class FeaturePipeline(Workflow):
    name = "feature"

    def build_phases(self):
        return [Planning, Implementation]


# --- Run ---

async def main():
    runner = DefaultWorkflowRunner(
        agent_runtime=ClaudeAgentRuntime(
            session_store=FileSessionStore("~/.iriai/sessions"),
        ),
        interaction_runtimes={
            "human": TerminalInteractionRuntime(),
            "auto": AutoApproveRuntime(),
        },
        artifacts=FileArtifactStore("~/.iriai/features"),
        context_provider=DefaultContextProvider(
            artifacts=FileArtifactStore("~/.iriai/features"),
            static_files={"project": Path("CLAUDE.md")},
        ),
        workspaces={"main": Workspace(id="main", path=Path("."), branch="main")},
    )

    feature = Feature(
        id="add-dark-mode",
        name="Add dark mode",
        slug="add-dark-mode",
        workflow_name="feature",
        workspace_id="main",
    )

    state = FeatureState(description="Add dark mode support to the application")
    await runner.execute_workflow(FeaturePipeline(), feature, state)
```

---

## Summary

This spec defines `iriai-compose` as a workflow composition library that is:

- **Actor-neutral** — tasks define interaction patterns between actors; actors can be agents, humans, or policies
- **Actor-stateful** — agents carry their own context and session identity; using the same actor twice continues the same conversation with the same worldview
- **Ergonomic** — phases read like scripts; actors defined once with context, tasks just say what to do
- **Runtime-agnostic** — core library never imports Claude SDK; adapters do
- **Storage-agnostic** — interfaces for artifacts, sessions, context; applications choose backends
- **Interaction-aware** — `Pending` as a core suspension primitive, resolved by pluggable runtimes
- **Pattern-rich** — `Interview` (multi-turn), `Gate` (approval), `Choose` (selection) are first-class
- **Extensible** — new interaction patterns (Debate, PanelReview) are just new Task subclasses with custom actor slots
- **Composable** — parallel via runner, child workflows via runner, phases own control flow
