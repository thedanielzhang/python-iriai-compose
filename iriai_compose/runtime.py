from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iriai_compose.tasks import Ask


class Runtime(ABC):
    """Base class for all runtimes.

    A runtime executes the atomic ``ask()`` operation.  Agent runtimes send
    prompts to LLMs; interaction runtimes present prompts to humans.

    ``ask()`` receives the full Ask task and decides how to use its fields
    (``prompt``, ``input``, ``input_type``, ``output_type``, ``to_prompt()``,
    etc.).  The framework passes ``context`` and runtime-specific metadata
    (``workspace``, ``session_key``) as keyword arguments.
    """

    name: str

    @abstractmethod
    async def ask(self, task: Ask, **kwargs: Any) -> Any:
        """Execute a single prompt→response interaction.

        Parameters
        ----------
        task:
            The Ask task.  The runtime reads whichever fields it needs
            (``task.prompt``, ``task.input``, ``task.to_prompt()``, etc.).
        **kwargs:
            Framework-level metadata:
            - ``context``: resolved context string (may be empty)
            - ``workspace``: the workspace for this feature (agent runtimes)
            - ``session_key``: session identifier (agent runtimes)
            Plus any runtime-specific hints passed via ``runner.run(..., **kwargs)``.
        """
        ...
