# iriai-sdk: Slack Bot Integration Guide

> Historical note
> This document references composite tasks like `Gate` and `Interview` as
> workflow patterns. Those composites are no longer exported by the public
> `iriai-compose` library API; current example implementations live under
> [`examples/_composite_tasks.py`](examples/_composite_tasks.py).

How `iriai-build` maps onto the `iriai-sdk` library, including the persistence layer.

This document uses `iriai-build` (the existing Slack-based agent orchestrator) as the reference application. It shows how each piece of the current system maps to the library's abstractions and where Postgres replaces the current SQLite + filesystem signal approach.

---

## Current Architecture (iriai-build)

For context, here's what exists today:

```
Slack (Socket Mode)
 ↕
slack-adapter.js          — message routing, Block Kit decisions
 ↕
orchestrator.js           — 3800-line state machine, feature lifecycle
 ↕
file-io.js                — chokidar watches signal directories
 ↕
.task / .done / .output   — filesystem signals for agent I/O
 ↕
agent-supervisor.js       — spawns claude CLI processes
 ↕
SQLite                    — features, agents, events, decisions tables
```

**What works:** The adapter pattern (Slack/Terminal/Desktop), the decision model, the event log, the agent hierarchy.

**What's painful:** The filesystem signal protocol is fragile (race conditions, cleanup failures, no atomicity). The orchestrator is a monolith. SQLite can't handle concurrent access from multiple processes well. State is split between SQLite rows and filesystem presence.

---

## Mapping to iriai-sdk

### The Decomposition

| iriai-build concept | iriai-sdk abstraction |
|---|---|
| `orchestrator.js` state machine | `Workflow` + `Phase` classes |
| `slack-adapter.js` decisions | `InteractionRuntime` (Slack impl) |
| `.task` → agent → `.done` | `AgentRuntime` (Claude impl) |
| `decisions` table rows | `Pending` |
| `events` table | Application-level event log (Postgres) |
| `features` table | `Feature` + application state |
| `agents` table | `SessionStore` + application process tracking |
| `file-io.js` signal watcher | Eliminated — `AgentRuntime` handles invocation directly |
| `agent-supervisor.js` | `AgentRuntime` internals (process management) |
| `operator.js` relay queue | Phase-level formatting logic |

### What Goes Away

The filesystem signal protocol (`.task`, `.done`, `.output`, `.active-task`, `.question`, `.answer`, `.gate-ready`, `.gate-approved`, `.gate-rejected`, `.crashed`, `.needs-restart`, `.feature-complete`, `.context-refresh`) is entirely replaced by:

- `runner.invoke()` for agent calls (direct SDK invocation, no file polling)
- `runner.request()` for human/external interaction (Pending → Slack → resolve)
- `runner.artifacts.put()` / `.get()` for artifact persistence

No more chokidar. No more atomic renames. No more `.done` polling loops.

---

## Persistence Layer: Postgres

### Why Postgres Over SQLite

- **Concurrent access.** Multiple features running simultaneously, each with parallel team agents. SQLite's write lock serializes everything.
- **Durable Pending resolution.** A Pending may sit unresolved for days. Postgres handles this naturally with row-level locking and transactional updates.
- **Process independence.** The runner, agent runtimes, and interaction runtimes can live in separate processes (or containers) sharing the same database.
- **Query flexibility.** Cross-feature analytics, pending dashboards, audit trails — all straightforward with Postgres.

### Schema

The schema mirrors the current SQLite tables but is designed around iriai-sdk's abstractions.

```sql
-- Features: execution instances
CREATE TABLE features (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE,
    workflow_name   TEXT NOT NULL,
    workspace_id    TEXT,
    phase           TEXT NOT NULL DEFAULT 'pending',
    phase_index     INT NOT NULL DEFAULT 0,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Pending: suspended interaction points
CREATE TABLE pendings (
    id              TEXT PRIMARY KEY,
    feature_id      TEXT NOT NULL REFERENCES features(id),
    phase_name      TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('approve', 'choose', 'respond')),
    prompt          TEXT NOT NULL,
    evidence        JSONB,
    options         JSONB,
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    response        JSONB,
    resolved_by     TEXT,                       -- slack user ID or "auto" or "agent"
    slack_ts        TEXT,                       -- message ts for Block Kit update
    slack_channel   TEXT,                       -- channel where decision was posted
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ
);
CREATE INDEX idx_pendings_unresolved ON pendings (feature_id) WHERE NOT resolved;

-- Agent sessions: runtime continuity
CREATE TABLE agent_sessions (
    session_key     TEXT PRIMARY KEY,
    feature_id      TEXT NOT NULL REFERENCES features(id),
    runtime         TEXT NOT NULL DEFAULT 'claude',
    session_id      TEXT,                       -- Claude SDK session ID
    model           TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Artifacts: versioned workflow outputs
CREATE TABLE artifacts (
    id              SERIAL PRIMARY KEY,
    feature_id      TEXT NOT NULL REFERENCES features(id),
    key             TEXT NOT NULL,
    content         JSONB NOT NULL,
    version         INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_artifacts_latest ON artifacts (feature_id, key, version);
CREATE INDEX idx_artifacts_lookup ON artifacts (feature_id, key);

-- Events: append-only audit log (replaces SQLite events table)
CREATE TABLE events (
    id              SERIAL PRIMARY KEY,
    feature_id      TEXT NOT NULL REFERENCES features(id),
    event_type      TEXT NOT NULL,
    source          TEXT,                       -- agent role, "user", "system"
    content         TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    slack_ts        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_events_feature ON events (feature_id, created_at);
CREATE INDEX idx_events_type ON events (feature_id, event_type);

-- Workspaces: registered workspace environments
CREATE TABLE workspaces (
    id              TEXT PRIMARY KEY,
    feature_id      TEXT REFERENCES features(id),
    path            TEXT NOT NULL,
    branch          TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### What Changed From Current SQLite

| Current SQLite table | Postgres table | Notes |
|---|---|---|
| `features` | `features` | Simpler. Phase tracking only. No signal_dir, no thread_ts (that's Slack metadata, lives in `metadata` JSONB). |
| `agents` | `agent_sessions` | No more PID tracking, retry counts, exit codes. Process management is inside `AgentRuntime`. Sessions only store what's needed for SDK resumption. |
| `events` | `events` | Identical purpose. Append-only audit log. |
| `decisions` | `pendings` | Direct mapping. `decision_type` → `kind`. `selected_option` → `response`. Slack-specific fields (ts, channel) kept for Block Kit updates. |
| `operator_relay_queue` | Eliminated | Operator formatting becomes a phase-level concern, not a queue. |
| `slack_posts` | Eliminated | Dedup handled by `pendings.slack_ts` and event idempotency. |

---

## Runtime Implementations

### PostgresArtifactStore

```python
class PostgresArtifactStore(ArtifactStore):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get(self, key: str, *, feature: Feature) -> Any | None:
        row = await self.pool.fetchrow("""
            SELECT content FROM artifacts
            WHERE feature_id = $1 AND key = $2
            ORDER BY version DESC LIMIT 1
        """, feature.id, key)
        return json.loads(row["content"]) if row else None

    async def put(self, key: str, value: Any, *, feature: Feature) -> None:
        # Get next version number
        current = await self.pool.fetchval("""
            SELECT COALESCE(MAX(version), 0) FROM artifacts
            WHERE feature_id = $1 AND key = $2
        """, feature.id, key)

        content = value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value)
        await self.pool.execute("""
            INSERT INTO artifacts (feature_id, key, content, version)
            VALUES ($1, $2, $3::jsonb, $4)
        """, feature.id, key, content, current + 1)
```

Artifacts are versioned. Every `put()` creates a new version. `get()` returns the latest. This replaces `SingleFile` (overwrite) and `VersionedFile` (keep history) — one implementation handles both needs.

### PostgresSessionStore

```python
class PostgresSessionStore(SessionStore):
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def load(self, session_key: str) -> AgentSession | None:
        row = await self.pool.fetchrow("""
            SELECT * FROM agent_sessions WHERE session_key = $1
        """, session_key)
        if not row:
            return None
        return AgentSession(
            session_key=row["session_key"],
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"]),
        )

    async def save(self, session: AgentSession) -> None:
        await self.pool.execute("""
            INSERT INTO agent_sessions (session_key, feature_id, runtime, session_id, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (session_key) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                metadata = EXCLUDED.metadata,
                updated_at = now()
        """, session.session_key, session.metadata.get("feature_id", ""),
             session.metadata.get("runtime", "claude"), session.session_id,
             json.dumps(session.metadata))
```

### SlackInteractionRuntime

This is the core of the Pending → Slack → resolve flow. Here's how it works end-to-end.

```python
class SlackInteractionRuntime(InteractionRuntime):
    """
    Resolves Pendings by posting Block Kit messages to Slack
    and waiting for user interaction (button clicks, reactions, modal submissions).
    """
    name = "slack"

    def __init__(self, slack_client, pool: asyncpg.Pool, default_channel: str):
        self.slack = slack_client          # @slack/web-api equivalent
        self.pool = pool
        self.default_channel = default_channel
        self._waiters: dict[str, asyncio.Future] = {}

    async def resolve(self, pending: Pending) -> str | bool:
        # 1. Persist the Pending to Postgres
        await self._persist_pending(pending)

        # 2. Post to Slack with Block Kit
        slack_ts = await self._post_to_slack(pending)

        # 3. Update the Pending row with Slack metadata
        await self.pool.execute("""
            UPDATE pendings SET slack_ts = $1, slack_channel = $2
            WHERE id = $3
        """, slack_ts, self._channel_for(pending), pending.id)

        # 4. Create a Future and park the coroutine
        future = asyncio.get_event_loop().create_future()
        self._waiters[pending.id] = future

        # 5. Await resolution — this is where the coroutine suspends
        response = await future

        return response

    async def _post_to_slack(self, pending: Pending) -> str:
        channel = self._channel_for(pending)

        if pending.kind == "approve":
            blocks = self._build_approval_blocks(pending)
        elif pending.kind == "choose":
            blocks = self._build_choice_blocks(pending)
        elif pending.kind == "respond":
            blocks = self._build_response_blocks(pending)

        result = await self.slack.chat_postMessage(
            channel=channel,
            blocks=blocks,
            text=pending.prompt,  # fallback
        )
        return result["ts"]

    # --- Called by Slack event handler when user interacts ---

    async def handle_slack_action(self, payload: dict):
        """
        Called by the Slack Socket Mode handler when a user clicks
        a button, submits a modal, or adds a reaction.
        """
        pending_id = payload["actions"][0]["value"]  # encoded in Block Kit
        action = payload["actions"][0]["action_id"]

        # Determine response
        if action == "approve":
            response = True
        elif action == "reject":
            # Open modal for feedback
            feedback = await self._collect_feedback_modal(payload)
            response = feedback if feedback else False
        elif action.startswith("choose:"):
            response = action.split(":", 1)[1]
        else:
            response = payload.get("text", "")

        # Persist resolution
        await self.pool.execute("""
            UPDATE pendings
            SET resolved = TRUE, response = $1::jsonb,
                resolved_by = $2, resolved_at = now()
            WHERE id = $3
        """, json.dumps(response), payload["user"]["id"], pending_id)

        # Update Slack message (remove buttons, show result)
        await self._update_slack_message(pending_id, response)

        # Wake up the waiting coroutine
        if pending_id in self._waiters:
            self._waiters[pending_id].set_result(response)
            del self._waiters[pending_id]

    # --- Recovery: reconnect waiters after process restart ---

    async def recover_pendings(self, runner: "WorkflowRunner"):
        """
        On startup, find unresolved Pendings in Postgres.
        For already-resolved ones (user responded while we were down),
        the workflow resumption logic handles them.
        For still-unresolved ones, re-register waiters.
        """
        rows = await self.pool.fetch("""
            SELECT * FROM pendings WHERE NOT resolved
        """)
        # These will be re-awaited when the workflow resumes
        # The Slack messages are still live — buttons still work
        # When user clicks, handle_slack_action fires as normal
        return [Pending(**row) for row in rows]
```

### The Pending Lifecycle (Slack)

Here's the full timeline of a Gate approval:

```
Phase code                         Postgres                    Slack                        asyncio
─────────                          ────────                    ─────                        ───────
Gate("Approve PRD?").run()
  ↓
runner.request("approve", ...)
  ↓
SlackInteractionRuntime.resolve()
  ├─ INSERT INTO pendings ────────→ Row: resolved=false
  ├─ chat.postMessage ──────────────────────────────────→ Block Kit message
  │                                                       [Approve] [Reject]
  ├─ future = create_future()
  └─ await future ─────────────────────────────────────────────────────────→ coroutine parked
                                                                             event loop free
                                                                             other features run

... hours pass ...

User clicks [Approve] in Slack
  ↓
Socket Mode event fires
  ↓
handle_slack_action()
  ├─ UPDATE pendings ────────────→ Row: resolved=true,
  │    SET resolved=true                response=true,
  │                                     resolved_by="U12345"
  ├─ chat.update ───────────────────────────────────────→ "✅ Approved by @daniel"
  └─ future.set_result(True) ──────────────────────────────────────────────→ coroutine wakes
                                                                             ↓
                                                                          Gate returns True
                                                                             ↓
                                                                          Phase continues
```

### Process Restart Recovery

What if the process dies while a Pending is outstanding?

```
Before crash:
  - Pending row in Postgres: resolved=false, slack_ts="1234.5678"
  - Slack message with buttons still visible in channel
  - asyncio Future is lost (in-memory only)

On restart:
  1. Application calls runner.recover()
  2. Loads all Features from Postgres where phase != 'complete'
  3. For each feature, loads its workflow and current phase_index
  4. Calls SlackInteractionRuntime.recover_pendings()
     - Finds unresolved Pendings in Postgres
  5. Re-executes the workflow from the current phase
     - Phase re-runs, hits the same Gate/Interview/Choose
     - runner.request() is called again
     - SlackInteractionRuntime.resolve() checks: "Is there already
       a Pending for this feature+phase+kind?"
       - If resolved (user clicked while we were down): return the stored response immediately
       - If unresolved: re-register a Future waiter, don't re-post to Slack
         (the original message with buttons is still live)

Scenario A: User responded while process was down
  - Slack webhook was lost, but user also clicked the button
  - Problem: Socket Mode doesn't deliver events when disconnected
  - Solution: On recovery, check if the Pending was resolved by
    polling Slack reactions/message state, OR
    rely on the Slack action being re-delivered when Socket Mode reconnects
    (Socket Mode has built-in retry for unacknowledged events)

Scenario B: User hasn't responded yet
  - Pending still unresolved in Postgres
  - Slack message still has live buttons
  - New Future registered, coroutine parks again
  - When user eventually clicks, everything works as normal
```

### ClaudeAgentRuntime (Replacing .task/.done)

The current filesystem signal protocol:
```
orchestrator writes .task → agent reads .task → agent writes .output + .done → orchestrator reads
```

Becomes a direct SDK call:

```python
class ClaudeAgentRuntime(AgentRuntime):
    name = "claude"

    def __init__(self, session_store: SessionStore | None = None):
        self.session_store = session_store

    async def invoke(self, role, prompt, *, output_type=None,
                     workspace=None, session_key=None):
        options = {
            "system_prompt": role.prompt,
            "allowed_tools": role.tools,
            "model": role.model or "claude-sonnet-4-6",
        }
        if workspace:
            options["cwd"] = str(workspace.path)
        if "setting_sources" in role.metadata:
            options["setting_sources"] = role.metadata["setting_sources"]

        # Resume existing session if persistent
        if session_key and self.session_store:
            session = await self.session_store.load(session_key)
            if session and session.session_id:
                options["resume"] = session.session_id

        if output_type:
            options["output_format"] = output_type.model_json_schema()

        # Direct SDK call — no filesystem intermediary
        async for msg in claude_sdk.query(prompt=prompt, options=options):
            if msg.type == "system" and msg.subtype == "init":
                if session_key and self.session_store:
                    await self.session_store.save(AgentSession(
                        session_key=session_key,
                        session_id=msg.session_id,
                    ))
            if msg.type == "result":
                if output_type:
                    return output_type.model_validate_json(msg.result)
                return msg.result
```

No `.task` file. No `.done` polling. No chokidar. No atomic rename race conditions. The SDK call blocks (async) until the agent finishes.

**Process management** (retries, health monitoring, memory limits) that currently lives in `agent-supervisor.js` can be handled by:
- Wrapping the SDK call with timeout + retry logic in the runtime
- The SDK itself managing subprocess lifecycle
- Application-level retry policies on task failure

---

## Workflow Mapping

### Current Orchestrator Phases → iriai-sdk Phases

The 3800-line orchestrator state machine decomposes into:

```python
class BuildFeatureWorkflow(Workflow):
    name = "build-feature"

    def build_phases(self):
        return [
            ScopingPhase,
            PMPhase,
            PMReviewPhase,
            DesignPhase,
            DesignReviewPhase,
            ArchitecturePhase,
            ArchitectureReviewPhase,
            PlanCompilerPhase,
            PlanApprovalPhase,
            ImplementationPhase,
            CompletionPhase,
        ]
```

Each phase is ~30-80 lines of clear orchestration logic instead of being tangled into a monolithic state machine.

### Example: PM Phase + Review (Current vs SDK)

**Current (orchestrator.js):**
```
1. Write .task to pm/ signal dir with YAML frontmatter
2. Watch for .done file via chokidar
3. Read .output file
4. Delete .done + .output
5. Insert event into SQLite
6. Write to operator relay queue
7. Spawn operator to format output
8. Watch for operator .done
9. Post formatted output to Slack
10. Post Block Kit decision (approve/reject)
11. Watch for decision resolution via Slack action handler
12. If rejected: write .user-message, respawn PM agent, goto 2
13. If approved: update feature phase, move to next role
```

**SDK version:**
```python
class PMPhase(Phase):
    name = "pm"

    async def execute(self, runner, feature, state):
        prd = await Ask(
            role=pm,
            prompt="Produce a PRD for this scoped feature.",
            context_keys=["project", "scoped-feature"],
            output_type=PRD,
            persistent=True,
        ).run(runner, feature)
        await runner.artifacts.put("prd", prd, feature=feature)
        return state


class PMReviewPhase(Phase):
    name = "pm-review"

    async def execute(self, runner, feature, state):
        result = await Gate(prompt="Approve PRD?").run(runner, feature)
        if isinstance(result, str):
            # Feedback string — revise
            prd = await Ask(
                role=pm,
                prompt=f"Revise the PRD using this feedback:\n\n{result}",
                context_keys=["project", "prd"],
                output_type=PRD,
                persistent=True,
            ).run(runner, feature)
            await runner.artifacts.put("prd", prd, feature=feature)
            # Re-gate
            await Gate(prompt="Approve revised PRD?").run(runner, feature)
        return state
```

13 steps of file I/O, signal watching, relay queuing, and state machine transitions become ~20 lines of declarative orchestration.

### Implementation Phase: Parallel Teams

**Current:** Feature Lead agent writes `.task` files to team orchestrator signal dirs, which write `.task` files to role agent signal dirs. Gate evidence flows back up via `.gate-ready` signals.

**SDK version:**
```python
class ImplementationPhase(Phase):
    name = "implementation"

    async def execute(self, runner, feature, state):
        plan = await runner.artifacts.get("plan", feature=feature)

        for phase_spec in plan.phases:
            # Fan out to teams in parallel
            team_tasks = []
            for team in phase_spec.teams:
                for task_spec in team.tasks:
                    team_tasks.append(Ask(
                        role=self._role_for(task_spec.role_name),
                        prompt=f"Implement: {task_spec.description}",
                        context_keys=["project", "plan", "architecture"],
                    ))
            results = await runner.parallel(team_tasks, feature)

            # Review
            review_tasks = [
                Ask(
                    role=integration_tester,
                    prompt="Test the implementation for this phase.",
                    context_keys=["project", "plan"],
                ),
                Ask(
                    role=code_reviewer,
                    prompt="Review code quality and correctness.",
                    context_keys=["project", "plan"],
                ),
            ]
            reviews = await runner.parallel(review_tasks, feature)
            await runner.artifacts.put(
                f"review-{phase_spec.name}", reviews, feature=feature
            )

            # Gate
            await Gate(
                prompt=f"Approve phase '{phase_spec.name}'?"
            ).run(runner, feature)

        return state
```

The Feature Lead agent, team orchestrator agents, and signal file choreography are all replaced by `runner.parallel()` and phase-level loops.

---

## Application Shell

The application that wraps iriai-sdk for the Slack bot:

```python
import asyncpg
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from iriai import (
    DefaultWorkflowRunner, Feature, Workspace,
    ClaudeAgentRuntime,
)


class IriaiBuildApp:
    """
    The application layer. Owns Postgres, Slack, process lifecycle,
    and crash recovery. Uses iriai-sdk for workflow definition and execution.
    """

    def __init__(self, pool: asyncpg.Pool, slack: SocketModeClient):
        self.pool = pool

        # Storage
        self.artifacts = PostgresArtifactStore(pool)
        self.sessions = PostgresSessionStore(pool)
        self.context_provider = PostgresContextProvider(
            artifacts=self.artifacts,
            static_files={"project": Path("CLAUDE.md")},
        )

        # Runtimes
        self.agent_runtime = ClaudeAgentRuntime(session_store=self.sessions)
        self.interaction_runtime = SlackInteractionRuntime(
            slack_client=slack.web_client,
            pool=pool,
            default_channel=PLANNING_CHANNEL,
        )

        # Runner
        self.runner = DefaultWorkflowRunner(
            agent_runtime=self.agent_runtime,
            interaction_runtime=self.interaction_runtime,
            artifacts=self.artifacts,
            sessions=self.sessions,
            context_provider=self.context_provider,
        )

        # Wire Slack actions to interaction runtime
        slack.socket_mode_request_listeners.append(self._handle_slack_event)

    async def start_feature(self, name: str, description: str, thread_ts: str):
        """Called when user posts [FEATURE] in Slack."""
        slug = slugify(name)

        # Create workspace (git worktree)
        workspace = await self._create_workspace(slug)

        # Persist feature
        feature = Feature(
            id=slug,
            name=name,
            slug=slug,
            workflow_name="build-feature",
            workspace_id=workspace.id,
            metadata={"thread_ts": thread_ts, "description": description},
        )
        await self._persist_feature(feature)
        self.runner.register_workspace(workspace)

        # Launch workflow as background task
        state = BuildFeatureState(description=description)
        asyncio.create_task(
            self._run_feature(feature, state)
        )

    async def _run_feature(self, feature: Feature, state: BaseModel):
        """Execute workflow, handle completion and errors."""
        try:
            workflow = BuildFeatureWorkflow()
            result = await self.runner.execute_workflow(workflow, feature, state)
            await self._complete_feature(feature)
        except Exception as e:
            await self._handle_feature_error(feature, e)

    async def recover(self):
        """On startup: resume in-progress features."""
        rows = await self.pool.fetch("""
            SELECT * FROM features WHERE phase != 'complete'
        """)
        for row in rows:
            feature = Feature(**row)
            state = await self._load_feature_state(feature)
            asyncio.create_task(
                self._run_feature(feature, state)
            )

    async def _handle_slack_event(self, req):
        """Route Slack interactive events to interaction runtime."""
        if req.type == "interactive":
            await self.interaction_runtime.handle_slack_action(req.payload)
```

### Startup

```python
async def main():
    pool = await asyncpg.create_pool(DATABASE_URL)
    slack = SocketModeClient(app_token=SLACK_APP_TOKEN, web_client=web_client)

    app = IriaiBuildApp(pool, slack)
    await app.recover()     # resume any in-progress features
    await slack.connect()   # start listening for Slack events

    # Keep alive
    await asyncio.Event().wait()
```

---

## Concurrency Model

Multiple features run concurrently as independent `asyncio.create_task()` coroutines.

```
Event Loop
 ├── Feature "dark-mode"     → Planning phase, awaiting Gate (Pending in Slack)
 ├── Feature "auth-refactor" → Implementation phase, parallel team agents running
 ├── Feature "fix-bug-123"   → PM Interview, awaiting human response
 ├── Slack Socket Mode       → Listening for button clicks, messages
 └── Recovery checker        → Periodic health check (optional)
```

When a feature hits `await runner.request(...)`:
1. Its coroutine suspends
2. Event loop continues serving other features and Slack events
3. When user clicks a Slack button, the Socket Mode handler fires
4. Handler calls `interaction_runtime.handle_slack_action()`
5. The Future resolves, the suspended coroutine wakes up
6. The feature's phase continues

No threads. No multiprocessing. No file polling. Just asyncio.

---

## What the Application Owns

| Concern | Where it lives | Not in iriai-sdk because |
|---|---|---|
| Postgres connection pool | `IriaiBuildApp.__init__` | Storage backend choice |
| Slack Socket Mode setup | `IriaiBuildApp.__init__` | Transport choice |
| Feature creation from Slack message | `start_feature()` | UI/trigger specific |
| Git worktree creation/cleanup | `_create_workspace()` | Deployment specific |
| Crash recovery orchestration | `recover()` | Application lifecycle |
| Error handling / retry policy | `_run_feature()` | Policy choice |
| Event logging | Postgres `events` table | Observability choice |
| Slack message formatting | `SlackInteractionRuntime` internals | Presentation choice |
| Budget tiers / model selection | Role definitions + metadata | Cost policy |
| Channel management | Application Slack code | Slack-specific |

## What iriai-sdk Owns

| Concern | Where it lives |
|---|---|
| Workflow/Phase/Task structure | Core library |
| `Ask`, `Interview`, `Gate`, `Choose`, `Respond` | Built-in tasks |
| `Pending` model | Core library |
| `runner.invoke()` → agent dispatch | `WorkflowRunner` + `AgentRuntime` |
| `runner.request()` → interaction dispatch | `WorkflowRunner` + `InteractionRuntime` |
| Context resolution (keys → prompt string) | `ContextProvider` |
| `runner.parallel()` | Orchestration helper |
| Phase sequencing | `execute_workflow()` |

The line is clean: iriai-sdk defines the workflow model and dispatch contracts. The application wires up Postgres, Slack, git, and process lifecycle.
