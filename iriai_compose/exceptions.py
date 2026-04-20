from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iriai_compose.tasks import Task
    from iriai_compose.workflow import Feature


class IriaiError(Exception):
    """Base exception for all iriai library errors."""

    pass


class ResolutionError(IriaiError):
    """Actor could not be routed to a runtime."""

    pass


class TaskExecutionError(IriaiError):
    """A task failed during execution.

    Wraps the underlying exception with context about which task,
    actors, phase, and feature were involved. The original exception
    is available via __cause__.
    """

    def __init__(self, *, task: Task, feature: Feature, phase_name: str):
        self.task = task
        self.feature = feature
        self.phase_name = phase_name
        actor_names = self._extract_actor_names(task)
        super().__init__(
            f"Task {type(task).__name__} failed in phase '{phase_name}' "
            f"for feature '{feature.id}' (actors: {actor_names})"
        )

    @staticmethod
    def _extract_actor_names(task: Task) -> str:
        names = []
        fields = getattr(type(task), "model_fields", {})
        for field_name in fields:
            actor = getattr(task, field_name, None)
            if hasattr(actor, "name"):
                names.append(actor.name)
            elif isinstance(actor, (list, tuple, set)):
                names.extend(item.name for item in actor if hasattr(item, "name"))
            elif isinstance(actor, dict):
                names.extend(item.name for item in actor.values() if hasattr(item, "name"))
        return ", ".join(names) or "unknown"
