from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class Pending(BaseModel):
    """A suspension point where the workflow is waiting on external input.

    .. deprecated::
        Runtimes now receive tasks directly via ``Runtime.ask()`` instead
        of Pending objects.  This class is kept for backwards compatibility
        with downstream consumers.
    """

    id: str
    feature_id: str
    phase_name: str
    kind: Literal["approve", "choose", "respond"]
    prompt: str
    evidence: Any | None = None
    options: list[str] | None = None
    created_at: datetime
    resolved: bool = False
    response: str | bool | None = None
