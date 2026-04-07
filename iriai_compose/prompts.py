"""Typed input models for Ask tasks.

These ``BaseModel`` subclasses serve as both *data carriers* (the ``input``
field on Ask) and *schema identifiers* (the ``input_type`` field on Ask).
Runtimes can inspect the type to decide presentation — for instance a
terminal runtime might show ``questionary.select()`` when it sees a
``Select`` input — but the framework does **not** prescribe this.
"""

from __future__ import annotations

from pydantic import BaseModel


class Select(BaseModel):
    """Selection from a list of options."""

    options: list[str]


class Confirm(BaseModel):
    """Yes/no confirmation."""

    pass
