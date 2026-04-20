# iriai-sdk: Agent Workflow Library Specification

> Historical note
> This spec still discusses `Interview`, `Gate`, and related interaction
> patterns for design context, but those composites are no longer exported by
> the public `iriai-compose` library API. Current example implementations live
> under [`examples/_composite_tasks.py`](examples/_composite_tasks.py) in the
> source tree and should be treated as application-owned patterns.

A Python library for composing agent workflows, built on the Claude Agent SDK.

---

## Design Principles

1. **Roles are lenses, not job descriptions.** A Role defines expertise and perspective — nothing about interaction pattern, output format, or orchestration behavior.
2. **Tasks are atomic.** The built-in task surface stays small: `Ask` reaches a runtime once, and coordination primitives like `Parallel` combine tasks without introducing new installed interaction types. Higher-level interview/gate flows are application-owned compositions, not built-in task primitives.
3. **Phases own control flow.** Branching, loops, retries, and conditional logic live in Phases, which compose Tasks.
4. **Workflows are templates.** A Workflow is a reusable sequence of Phases. A Feature is a specific execution of a Workflow with its own workspace and state.
5. **Context is injected, not owned.** No object owns context. A ContextProvider resolves keys to content, scoped to the Task (not the agent).
6. **Human interaction is a first-class runner primitive.** `request()` is fundamentally different from `invoke()` — it suspends the workflow and produces a `Pending` that can be inspected, persisted, and resolved externally. Application-owned interview/gate helpers can build on `request()` without becoming part of the installed task API.

---

## Core Abstractions

### Hierarchy

```
Iriai               — runtime             — manages all features
Feature             — instance            — one piece of work (name + state + workspace)
Workflow            — template            — reusable phase sequence
Phase               — task group          — tasks + control flow (branching, loops)
Task                — atomic operation    — Ask (+ app-owned composites)
Workspace           — physical context    — worktrees, branches, artifacts
Role                — expertise           — agent config (prompt, tools, model)
Agent               — running instance    — session state, resumable
ContextStore        — registry            — maps keys to (type, storage) pairs
ContextProvider     — resolver            — assembles context string from keys
Pending             — suspended state     — what's waiting on a human
FeatureState        — serializable        — survives crashes
```

### Ownership

```
Iriai
 ├── ContextStore (shared across features)
 ├── ContextProvider (default, overridable per feature/phase)
 └── Feature (many, concurrent)
      ├── Workflow (one, shared template)
      │    └── Phase (sequential)
      │         └── Task (atomic)
      ├── Workspace (one primary + N team worktrees)
      └── FeatureState (serialized to disk)
           ├── Pending actions
           └── Agent session IDs
```

---

## Role

Pure expertise. No interaction logic, no output format, no orchestration instructions.

A Role is a **lens** — it defines who the agent is and what it knows. Everything about how it's used (interview vs one-shot, what to output, what context to include) is determined by the Task and Phase that invoke it.

```python
class Role:
    name: str                              # identity
    prompt: str                            # expertise/perspective (2-3 sentences)
    tools: list[str]                       # capabilities
    model: str = "claude-opus-4-6"         # which claude model
    setting_sources: list[str] = ["project"]  # CLAUDE.md loading
```

**Guidelines for Role prompts:**
- Describe expertise and perspective only
- No "you MUST" orchestration instructions — that's the Task's job
- No output format instructions — that's the Task's `output_type` parameter
- No interaction pattern instructions — that's the Phase's control flow

**Example:**
```python
pm = Role(
    name="pm",
    prompt="You are a Product Manager. You understand user needs, business "
           "value, and requirement clarity. You identify gaps and ambiguities "
           "in specifications.",
    tools=["Read", "Glob", "Grep"],
)
```

The same Role can be used in different contexts without modification:
- As an interviewer (Phase drives the conversation loop)
- As a one-shot analyst (Phase calls Ask once)
- As a reviewer (Phase passes implementation + spec, asks for a Verdict)

---

## Agent

A running instance of a Role. Holds session state.

```python
class Agent:
    role: Role
    persistent: bool                       # whether to store session_id
    context: ContextPolicy | None          # context window management
    session_id: str | None                 # for resuming sessions
```

**Methods:**

```python
async def ask(self, prompt: str, output_type: type[BaseModel] | None = None) -> str | BaseModel
    """One-shot. If persistent and session exists, resumes. Otherwise starts fresh."""

async def followup(self, prompt: str, output_type: type[BaseModel] | None = None) -> str | BaseModel
    """Continue the conversation. Requires an existing session."""

async def checkpoint(self) -> str
    """Ask the agent to summarize its current state for handover."""
```

**Persistence behavior:**
- `persistent=True`: session_id is stored after first invocation. Subsequent calls resume the session. The agent remembers prior conversation.
- `persistent=False`: every call starts fresh. No session stored. Default.

**Creating agents:**
- `Agent(role)` — new instance, fresh context
- `agent.ask(...)` — use the agent (starts or resumes session)
- Tasks create agents via `self.agent(role, persistent=...)` which scopes them to the task

---

## ContextPolicy

Controls context window management (when to checkpoint/restart). Separate from knowledge context.

```python
class ContextPolicy:
    checkpoint_threshold: float = 0.4      # restart when 40% context remaining
```

Attached to an Agent at creation. The runtime monitors context usage and triggers checkpoint + restart when the threshold is reached.

---

## Task

Atomic operations. Each Task does exactly one thing.

### Base

```python
class Task:
    role: Role | None                      # which role to use (if agent-based)
    context_keys: list[str]                # what context this task needs
    workspace: Workspace                   # injected by Phase
    context_provider: ContextProvider      # injected by Phase

    def agent(self, role: Role, persistent=False,
              context: ContextPolicy | None = None) -> Agent:
        """Get or create an agent scoped to this task."""

    async def invoke(self, agent: Agent, prompt: str,
                     output_type: type[BaseModel] | None = None) -> str | BaseModel:
        """Call an agent. Assembles context from context_keys, prepends to prompt.
        Returns when agent finishes. This is a programmatic function call."""

    async def request(self, kind: str, prompt: str, evidence=None,
                      options: list[str] | None = None) -> str | bool:
        """Suspend workflow. Returns when human responds.
        This is fundamentally different from invoke() — it creates a Pending
        and the workflow halts until external resolution."""
```

### Built-in Task Types

```python
class Ask(Task):
    """One-shot agent invocation. Send prompt, get typed result."""
    def __init__(self, role: Role, context: list[str] | None = None):
        ...
    async def run(self, prompt: str, output_type=None) -> str | BaseModel:
        ...

class Parallel(Task):
    """Run multiple tasks concurrently."""
    def __init__(self, tasks: list[Task]):
        ...
    async def run(self, input) -> list:
        return await asyncio.gather(*[t.run(input) for t in self.tasks])
```

### Example Composite Task Patterns (application-owned)

These patterns used to ship as public library tasks, but in `0.3.0` they moved
out of the installed API. They remain useful reference designs for application
layers that want richer human-interaction workflows on top of `Ask`.

```python
class Interview(Task):
    """Multi-turn: agent asks questions, human answers, loop until
    agent produces typed output with no remaining questions."""
    def __init__(self, role: Role, context: list[str] | None = None):
        ...
    async def run(self, initial_prompt: str, output_type=None) -> BaseModel:
        agent = self.agent(self.role, persistent=True)
        response = await self.invoke(agent, initial_prompt)
        while True:
            answer = await self.request("respond", response)
            result = await self.invoke(agent, f"User says: {answer}", output_type=output_type)
            if isinstance(result, output_type) and not getattr(result, 'questions', []):
                return result
            response = result.questions[0] if hasattr(result, 'questions') else str(result)

class Gate(Task):
    """Human approval. Returns True, False, or feedback string."""
    def __init__(self, prompt: str):
        ...
    async def run(self, evidence=None) -> bool | str:
        return await self.request("approve", self.prompt, evidence=evidence)

class Choose(Task):
    """Human selects from options."""
    def __init__(self, prompt: str, options: list[str]):
        ...
    async def run(self, context=None) -> str:
        return await self.request("choose", self.prompt, evidence=context, options=self.options)
```

### Three Primitives

Every Task is built from three fundamental operations:

| Primitive | What happens | Who drives it | Duration |
|-----------|-------------|---------------|----------|
| `invoke(agent, prompt)` | Call an agent, get result | Programmatic | Seconds–minutes |
| `request(kind, prompt)` | Suspend, wait for human | Human | Minutes–days |
| `await asyncio.gather()` | Run tasks concurrently | Runtime | Varies |

---

## Phase

Groups Tasks with control flow. This is where branching, loops, retries, and conditional logic live.

```python
class Phase:
    name: str
    context_provider: ContextProvider | None   # override, inherits from Feature if None

    def agent(self, role: Role, persistent=False,
              context: ContextPolicy | None = None) -> Agent:
        """Get or create an agent scoped to this phase."""

    async def run(self, input: BaseModel, workspace: Workspace,
                  store: ContextStore) -> BaseModel:
        """Execute the phase. Subclasses implement this."""
        raise NotImplementedError
```

**Example using only current primitives (`Ask` + `request()`):**

```python
class Planning(Phase):
    async def run(self, scoping: ScopingResult, workspace, store) -> Plan:
        # One-shot ask for an initial PRD draft
        prd = await Ask(pm, context=["project"]).run(
            f"Analyze this scope and draft a PRD.\n{scoping}",
            output_type=PRD,
        )
        await store.put("prd", self.feature, prd)

        # Suspend for human approval using the built-in request primitive
        prd_feedback = await self.request("approve", "Approve PRD?", evidence=prd)
        if prd_feedback is not True:
            prd = await Ask(pm, context=["project"]).run(
                f"Revise the PRD using this feedback:\n{prd_feedback}",
                output_type=PRD,
            )
            await store.put("prd", self.feature, prd)

        # One-shot — agent produces output directly
        design = await Ask(designer, context=["project", "prd"]).run(
            f"Produce design decisions", output_type=DesignDecisions
        )
        await store.put("design", self.feature, design)

        # Suspend for a human choice if the designer proposes alternatives
        if design.alternatives:
            choice = await self.request(
                "choose",
                "Which approach?",
                evidence=design,
                options=design.alternatives,
            )
            design = await Ask(designer, context=["project", "prd"]).run(
                f"Finalize with approach: {choice}", output_type=DesignDecisions
            )
        design_feedback = await self.request("approve", "Approve design?", evidence=design)
        if design_feedback is not True:
            design = await Ask(designer, context=["project", "prd"]).run(
                f"Revise the design using this feedback:\n{design_feedback}",
                output_type=DesignDecisions,
            )
            await store.put("design", self.feature, design)

        # Produce a technical plan directly
        plan = await Ask(architect, context=["project", "prd", "design"]).run(
            "Produce a technical plan for implementation.",
            output_type=Plan,
        )

        # Loop — validate until plan compiles
        verdict = await Ask(plan_compiler, context=["project"]).run(
            f"Validate this plan", output_type=Verdict
        )
        while not verdict.passed:
            plan = await Ask(architect, context=["project", "prd", "design"]).run(
                f"Fix these issues: {verdict.issues}", output_type=Plan
            )
            verdict = await Ask(plan_compiler, context=["project"]).run(
                f"Validate this plan", output_type=Verdict
            )

        await store.put("plan", self.feature, plan)
        plan_feedback = await self.request(
            "approve",
            "Approve plan for implementation?",
            evidence=plan,
        )
        if plan_feedback is not True:
            plan = await Ask(architect, context=["project", "prd", "design"]).run(
                f"Revise the plan using this feedback:\n{plan_feedback}",
                output_type=Plan,
            )
            await store.put("plan", self.feature, plan)
        return plan
```

---

## Workflow

A reusable sequence of Phases. Stateless template — does not own execution state.

```python
class Workflow:
    phases: list[type[Phase]]

    async def run(self, initial_input: BaseModel, workspace: Workspace,
                  store: ContextStore, state: FeatureState) -> BaseModel:
        """Execute phases in sequence. Output of each feeds into the next."""
        result = initial_input
        for phase_cls in self.phases:
            phase = phase_cls()
            result = await phase.run(result, workspace, store)
            state.complete_phase(phase.name, result)
            state.save()
        return result

    async def resume(self, state: FeatureState, workspace: Workspace,
                     store: ContextStore) -> BaseModel:
        """Resume from where we left off after crash."""
        ...
```

**Example:**
```python
feature_pipeline = Workflow(phases=[Scoping, Planning, Implementation])
hotfix_pipeline = Workflow(phases=[HotfixPlanning, HotfixImplementation])
```

---

## Feature

A specific piece of work. Binds a Workflow (template) to a Workspace (physical context) and FeatureState (execution state).

```python
class Feature:
    name: str
    slug: str
    workflow: Workflow
    workspace: Workspace
    state: FeatureState
    context_provider: ContextProvider | None   # override

    @classmethod
    def create(cls, name: str, workflow: Workflow, codebase: Path,
               context_provider: ContextProvider | None = None) -> Feature:
        slug = slugify(name)
        workspace = Workspace.create(codebase, branch=f"feat/{slug}")
        return cls(
            name=name, slug=slug, workflow=workflow,
            workspace=workspace, state=FeatureState(),
            context_provider=context_provider,
        )

    async def start(self):
        """Begin executing the workflow."""
        ...

    async def resume(self):
        """Resume after crash or restart."""
        ...
```

---

## Workspace

A physical place where agents do work. Manages git worktrees for team isolation.

```python
class Workspace:
    codebase: Path                         # repo root
    branch: str                            # git branch
    worktree: Path | None                  # isolated copy (for teams)
    artifacts: Path                        # where artifacts live on disk

    @classmethod
    def create(cls, codebase: Path, branch: str) -> Workspace:
        """Create a git worktree for isolated work."""
        ...

    @classmethod
    def shared(cls, codebase: Path) -> Workspace:
        """No isolation. Work directly on the repo."""
        ...

    @property
    def path(self) -> Path:
        """The directory agents should work in (cwd)."""
        return self.worktree or self.codebase

    def merge_to(self, target: Workspace):
        """Merge this workspace's branch into target."""
        ...

    def diff_summary(self) -> str:
        """Summary of changes on this branch."""
        ...

    def cleanup(self):
        """Remove the worktree."""
        ...
```

**Usage in Phases:**
```python
class Implementation(Phase):
    async def run(self, plan, workspace, store):
        # Create isolated workspaces per team
        team_workspaces = [
            Workspace.create(workspace.codebase, branch=f"feat/{slug}-team-{i}")
            for i in range(plan.team_count)
        ]

        # Tasks run in team workspaces
        results = await Parallel([
            Ask(backend_impl, context=["project", "plan"]).with_workspace(team_workspaces[i])
            for i in range(plan.team_count)
        ]).run(plan)

        # Merge after gate approval
        for ws in team_workspaces:
            ws.merge_to(workspace)
            ws.cleanup()
```

Agents receive `workspace.path` as their `cwd`. The Role does not define `cwd` — the Workspace does.

---

## Pending

A suspended point in the workflow where human input is required.

```python
class Pending:
    id: str                                # unique identifier
    feature_slug: str                      # which feature
    phase_name: str                        # which phase
    kind: str                              # "approve" | "choose" | "respond"
    prompt: str                            # what the user sees
    evidence: BaseModel | str | None       # context for the decision
    options: list[str] | None              # for "choose" kind
    created_at: datetime
    resolved: bool = False
    response: str | bool | None = None

    def resolve(self, input: str | bool):
        """User provides input. Workflow resumes."""
        self.resolved = True
        self.response = input
```

**Pending is serializable.** It survives process restarts. Multiple Pendings can exist simultaneously (parallel phases waiting on different gates).

---

## FeatureState

Serializable execution state. Persisted to disk. Enables crash recovery.

```python
class FeatureState:
    current_phase: str
    phase_index: int
    pending: list[Pending]
    completed_phases: dict[str, BaseModel]   # phase_name -> output
    agent_sessions: dict[str, str]           # agent_key -> session_id
    created_at: datetime
    updated_at: datetime

    def save(self, path: Path): ...
    @classmethod
    def load(cls, path: Path) -> FeatureState: ...
```

---

## Context System

### Overview

Context is **task-scoped** — the Task declares what context it needs via `context` keys. The ContextProvider resolves keys to content. The ContextStore maps keys to (type, storage) pairs.

Context is NOT agent-scoped. A code reviewer and security auditor doing the same review task get the same context, because the **work** determines the context, not the **worker**.

### ContextType

Defines what kind of document a context entry is and how to render it for agents.

```python
class ContextType:
    name: str                              # "artifact", "log", "static", "computed"
    format: str                            # "markdown" | "json" | "jsonl" | "yaml"
    schema: type[BaseModel] | None         # optional typed schema
    render: Callable[[str], str]           # how to turn raw content into agent-readable text
```

### StorageBackend

Defines how a context entry is persisted.

```python
class StorageBackend:
    async def read(self, key: str, feature: Feature) -> str | None: ...
    async def write(self, key: str, feature: Feature, content: str) -> None: ...
```

**Built-in backends:**

| Backend | Behavior | Use case |
|---------|----------|----------|
| `SingleFile` | One file per key. Overwritten on each write. | PRD, design, plan — any document |
| `AppendLog` | Each write appends an entry. Read returns all. | Task outputs, changelog, review history |
| `ReadOnly` | Reads static files. Writes are no-ops. | CLAUDE.md, GOTCHAS.md, project docs |
| `Ephemeral` | Generated by function on read. Never stored. | Cross-feature summaries, git state, APIs |
| `VersionedFile` | Keeps every version. Read returns latest. | Plans that evolve through revisions |

### ContextStore

Registry mapping keys to (type, storage) pairs.

```python
class ContextStore:
    default_storage: StorageBackend        # for unregistered keys

    def define(self, key: str, type: ContextType, storage: StorageBackend):
        """Register a context key with its type and storage."""
        ...

    async def get(self, key: str, feature: Feature) -> str | None:
        """Read content. Resolves type and storage from registry."""
        ...

    async def put(self, key: str, feature: Feature, content: str | BaseModel):
        """Write content. Resolves type and storage from registry."""
        ...
```

**Configuration:**
```python
# Type definitions (reusable)
artifact  = ContextType(name="artifact",  format="markdown", render=str)
log       = ContextType(name="log",       format="jsonl",    render=render_log_entries)
static    = ContextType(name="static",    format="markdown", render=str)
computed  = ContextType(name="computed",   format="markdown", render=str)

# Key -> (type, storage) mapping
store = ContextStore(default_storage=SingleFile(features_dir))
store.define("project",        static,    ReadOnly(codebase / "CLAUDE.md", codebase / "GOTCHAS.md"))
store.define("prd",            artifact,  SingleFile(features_dir))
store.define("design",         artifact,  SingleFile(features_dir))
store.define("plan",           artifact,  SingleFile(features_dir))
store.define("outputs",        log,       AppendLog(features_dir))
store.define("cross-feature",  computed,  Ephemeral(build_cross_feature_summary))
```

**Adding new document types requires no code changes:**
```python
store.define("threat-model",      artifact, SingleFile(features_dir))
store.define("api-contract",      artifact, SingleFile(features_dir))
store.define("review-history",    log,      AppendLog(features_dir))
store.define("deploy-status",     computed, Ephemeral(check_deploy_status))
```

### ContextProvider

Resolves context keys to a single string for agent prompts. Handles token budgeting.

```python
class ContextProvider:
    store: ContextStore

    async def resolve(self, keys: list[str], feature: Feature,
                      budget_tokens: int | None = None) -> str:
        """Resolve keys to context string. Earlier keys = higher priority
        when budget is tight."""
        ...
```

### Context Cascading

Context providers cascade: Task checks Phase, Phase checks Feature, Feature checks Iriai. First one with a provider wins.

```python
# Project-wide default
iriai = Iriai(codebase=..., context_provider=SiloedProvider())

# Override per feature
feature = Feature.create("auth refactor", context_provider=SharedProvider())

# Override per phase
class Implementation(Phase):
    context_provider = SharedProvider()   # impl sees cross-feature changes
```

### Context Scoping

Features control which context keys are available:

```python
# Siloed: feature only sees its own artifacts
dark_mode = Feature.create("dark mode", allowed_context=["project", "prd", "plan", "design"])

# Shared: feature sees cross-feature context
auth_refactor = Feature.create("auth refactor", allowed_context=["project", "prd", "plan", "design", "cross-feature"])
```

---

## Iriai (Runtime)

Top-level runtime. Manages all features, resolves pending actions, handles crash recovery.

```python
class Iriai:
    codebase: Path
    features: dict[str, Feature]
    store: ContextStore
    context_provider: ContextProvider       # default

    async def new_feature(self, name: str, workflow: Workflow,
                          context_provider: ContextProvider | None = None):
        """Create and start a new feature."""
        feature = Feature.create(name, workflow, self.codebase, context_provider)
        self.features[feature.slug] = feature
        await feature.start()

    def pending(self) -> list[Pending]:
        """All pending actions across all features."""
        return [p for f in self.features.values() for p in f.state.pending]

    async def resolve(self, pending_id: str, input: str | bool):
        """User responds to a pending action. Workflow resumes."""
        ...

    async def recover(self):
        """After crash: reload all features from persisted state, resume."""
        ...
```

---

## On-Disk Structure

```
~/.iriai/
  config.json                              # global config
  features/
    add-dark-mode/
      state.json                           # FeatureState
      artifacts/
        prd.json                           # typed (Pydantic)
        prd.md                             # agent-readable
        design.json
        design.md
        plan.json
        plan.md
      outputs.jsonl                        # append-only task completion log
      workspaces/
        team-0/                            # git worktree
        team-1/                            # git worktree
    fix-auth-bypass/
      state.json
      artifacts/
        ...
```

---

## Execution Model

### Agent Invocations

Every agent invocation flows through the Claude Agent SDK:

```python
async def invoke(self, agent: Agent, prompt: str, output_type=None):
    # 1. Resolve context keys to content
    context = await self.context_provider.resolve(
        self.context_keys, self.feature
    )

    # 2. Assemble full prompt
    full_prompt = f"{context}\n\n## Task\n{prompt}" if context else prompt

    # 3. Build SDK options
    options = ClaudeAgentOptions(
        system_prompt=agent.role.prompt,
        allowed_tools=agent.role.tools,
        model=agent.role.model,
        cwd=self.workspace.path,
        setting_sources=agent.role.setting_sources,
    )
    if agent.persistent and agent.session_id:
        options.resume = agent.session_id
    if output_type:
        options.output_format = output_type.model_json_schema()

    # 4. Call Agent SDK
    async for msg in query(prompt=full_prompt, options=options):
        if isinstance(msg, SystemMessage) and msg.subtype == "init":
            if agent.persistent:
                agent.session_id = msg.session_id
        if isinstance(msg, ResultMessage):
            if output_type:
                return output_type.model_validate_json(msg.result)
            return msg.result
```

### Human Interactions

`request()` creates a `Pending`, persists workflow state, and suspends:

```python
async def request(self, kind, prompt, evidence=None, options=None):
    pending = Pending(
        kind=kind, prompt=prompt, evidence=evidence, options=options,
        feature_slug=self.feature.slug, phase_name=self.phase.name,
    )
    self.feature.state.pending.append(pending)
    self.feature.state.save()

    # Suspend until resolved (by Slack callback, terminal input, API call, etc.)
    await pending.wait()

    return pending.response
```

### Crash Recovery

1. On startup, `Iriai.recover()` scans `~/.iriai/features/` for `state.json` files
2. Loads FeatureState for each feature
3. Checks for unresolved Pendings (re-present to user)
4. Resumes Workflow from the current phase using completed_phases as prior output
5. Agent sessions are resumed via stored session_ids

---

## Full Pipeline Example

This end-to-end sketch uses only the current installed primitives. If an
application wants richer interview/gate helpers, it can wrap the repeated
`request()` and `Ask(...)` patterns in local composite tasks.

```python
# --- Roles ---
pm = Role(name="pm", prompt="You understand user needs and requirement clarity.", tools=["Read", "Glob", "Grep"])
designer = Role(name="designer", prompt="You understand user flows and component design.", tools=["Read", "Glob", "Grep"])
architect = Role(name="architect", prompt="You understand system design and dependency graphs.", tools=["Read", "Glob", "Grep", "Bash"])
backend = Role(name="backend", prompt="You know FastAPI, SQLAlchemy, and async Python.", tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"])
reviewer = Role(name="reviewer", prompt="You evaluate correctness and adherence to spec.", tools=["Read", "Bash", "Glob", "Grep"])

# --- Phases ---
class Planning(Phase):
    async def run(self, scoping, workspace, store):
        prd = await Ask(pm, context=["project"]).run(f"Analyze: {scoping}", output_type=PRD)
        await store.put("prd", self.feature, prd)
        prd_feedback = await self.request("approve", "Approve PRD?", evidence=prd)
        if prd_feedback is not True:
            prd = await Ask(pm, context=["project"]).run(
                f"Revise PRD with feedback: {prd_feedback}",
                output_type=PRD,
            )
            await store.put("prd", self.feature, prd)

        design = await Ask(designer, context=["project", "prd"]).run("Produce design", output_type=Design)
        await store.put("design", self.feature, design)
        if design.alternatives:
            choice = await self.request(
                "choose",
                "Which design approach should we take?",
                evidence=design,
                options=design.alternatives,
            )
            design = await Ask(designer, context=["project", "prd"]).run(
                f"Finalize with approach: {choice}",
                output_type=Design,
            )
            await store.put("design", self.feature, design)
        design_feedback = await self.request("approve", "Approve design?", evidence=design)
        if design_feedback is not True:
            design = await Ask(designer, context=["project", "prd"]).run(
                f"Revise design with feedback: {design_feedback}",
                output_type=Design,
            )
            await store.put("design", self.feature, design)

        plan = await Ask(architect, context=["project", "prd", "design"]).run("Plan this", output_type=Plan)
        await store.put("plan", self.feature, plan)
        plan_feedback = await self.request("approve", "Approve plan?", evidence=plan)
        if plan_feedback is not True:
            plan = await Ask(architect, context=["project", "prd", "design"]).run(
                f"Revise plan with feedback: {plan_feedback}",
                output_type=Plan,
            )
            await store.put("plan", self.feature, plan)

        return plan

class Implementation(Phase):
    async def run(self, plan, workspace, store):
        for phase_spec in plan.phases:
            results = await Parallel([
                Ask(backend, context=["project", "plan"]) for _ in phase_spec.tasks
            ]).run(phase_spec)

            verdicts = await Parallel([
                Ask(reviewer, context=["project", "plan", "implementation"]),
            ]).run(results)

            phase_feedback = await self.request(
                "approve",
                f"Approve {phase_spec.name}?",
                evidence=verdicts,
            )
            if phase_feedback is not True:
                raise RuntimeError(
                    f"Implementation phase {phase_spec.name} requires revision: {phase_feedback}"
                )

        return ImplementationResult(...)

# --- Workflow ---
feature_pipeline = Workflow(phases=[Planning, Implementation])

# --- Run ---
iriai = Iriai(codebase=Path("/Users/danielzhang/src/iriai"))
await iriai.new_feature("Add dark mode", feature_pipeline)
```
