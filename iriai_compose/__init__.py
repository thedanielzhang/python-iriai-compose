"""iriai-compose: Runtime-agnostic, actor-neutral workflow orchestration."""

from iriai_compose.actors import Actor, AgentActor, InteractionActor, Role
from iriai_compose.exceptions import IriaiError, ResolutionError, TaskExecutionError
from iriai_compose.pending import Pending
from iriai_compose.prompts import Confirm, Select
from iriai_compose.runner import (
    AgentRuntime,
    DefaultWorkflowRunner,
    InteractionRuntime,
    WorkflowRunner,
)
from iriai_compose.runtime import Runtime
from iriai_compose.storage import (
    AgentSession,
    ArtifactStore,
    ContextProvider,
    DefaultContextProvider,
    InMemoryArtifactStore,
    InMemorySessionStore,
    InMemoryStore,
    SessionStore,
    Store,
)
from iriai_compose.tasks import Ask, Choose, Gate, Interview, Respond, Task, to_str
from iriai_compose.workflow import Feature, Phase, Workflow, Workspace

__all__ = [
    # actors
    "Actor",
    "AgentActor",
    "InteractionActor",
    "Role",
    # tasks
    "Task",
    "Ask",
    "Interview",
    "Gate",
    "Choose",
    "Respond",
    "to_str",
    # prompts (input types)
    "Select",
    "Confirm",
    # workflow
    "Phase",
    "Workflow",
    "Feature",
    "Workspace",
    # runner
    "WorkflowRunner",
    "DefaultWorkflowRunner",
    "Runtime",
    "AgentRuntime",
    "InteractionRuntime",
    # pending (deprecated)
    "Pending",
    # storage
    "Store",
    "InMemoryStore",
    "ArtifactStore",
    "SessionStore",
    "AgentSession",
    "ContextProvider",
    "InMemoryArtifactStore",
    "InMemorySessionStore",
    "DefaultContextProvider",
    # exceptions
    "IriaiError",
    "ResolutionError",
    "TaskExecutionError",
]
