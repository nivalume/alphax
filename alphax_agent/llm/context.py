"""What a model call is being made for.

The workflow, the coder and the agents describe their work with :func:`call_context` (loop, stage, task,
attempt, conversation, turn). The audit log reads :func:`current_context` when it records a call, so every line
of ``llm_calls.jsonl`` says which loop and stage it belongs to without the LLM client knowing about either.

The context lives in a :class:`~contextvars.ContextVar`, so it follows the code into LangGraph's worker threads
(which run in a copy of the context of the code that scheduled them).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_CONTEXT: ContextVar[Mapping[str, Any] | None] = ContextVar("alphax_call_context", default=None)


@contextmanager
def call_context(**fields: Any) -> Iterator[None]:
    """Add ``fields`` to the context of every model call made inside the block."""
    token = _CONTEXT.set({**(_CONTEXT.get() or {}), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_context() -> dict[str, Any]:
    return dict(_CONTEXT.get() or {})
