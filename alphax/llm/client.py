"""The LLM client every agent talks to."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alphax.config.settings import LLMSettings
from alphax.core.exceptions import LLMTransientError
from alphax.llm.backend import ChatBackend, LiteLLMBackend
from alphax.llm.cache import ResponseCache
from alphax.llm.types import ChatMessage, ChatRequest, ChatResponse, TokenUsage

logger = logging.getLogger(__name__)


class LLMAuditLog:
    """Append-only JSONL record of every model call: who asked, what was sent, what came back."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, request: ChatRequest, response: ChatResponse | None, seconds: float, error: str | None) -> None:
        entry: dict[str, Any] = {
            "time": datetime.now(UTC).isoformat(),
            "agent": request.agent,
            "seconds": round(seconds, 3),
            "request": request.model_dump(mode="json"),
            "response": None if response is None else response.model_dump(mode="json"),
            "error": error,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class LLMClient:
    """Adds retries, caching, usage accounting and auditing on top of a :class:`ChatBackend`.

    The client is shared by all agents of a run and is safe to use from several threads.
    """

    def __init__(
        self,
        backend: ChatBackend,
        settings: LLMSettings | None = None,
        *,
        cache: ResponseCache | None = None,
        audit: LLMAuditLog | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._backend = backend
        self._settings = settings or LLMSettings(model=backend.model)
        self._cache = cache
        self._audit = audit
        self._sleep = sleeper
        self._usage = TokenUsage()
        self._usage_by_agent: dict[str, TokenUsage] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: LLMSettings, audit_path: Path | None = None) -> LLMClient:
        cache = ResponseCache(settings.cache_dir) if settings.cache_dir is not None else None
        audit = LLMAuditLog(audit_path) if audit_path is not None else None
        return cls(LiteLLMBackend(settings), settings, cache=cache, audit=audit)

    @property
    def model(self) -> str:
        return self._backend.model

    @property
    def usage(self) -> TokenUsage:
        with self._lock:
            return self._usage.model_copy()

    def usage_by_agent(self) -> dict[str, TokenUsage]:
        with self._lock:
            return {agent: usage.model_copy() for agent, usage in self._usage_by_agent.items()}

    def attach_audit(self, audit: LLMAuditLog) -> None:
        self._audit = audit

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        agent: str = "anonymous",
    ) -> ChatResponse:
        request = ChatRequest(messages=list(messages), tools=tools or None, json_mode=json_mode, agent=agent)
        cache = self._cache
        key = request.cache_key(self.model, self._settings.temperature) if cache is not None else None
        if cache is not None and key is not None and (cached := cache.get(key)) is not None:
            self._record(request, cached, 0.0, None)
            return cached
        started = time.monotonic()
        try:
            response = self._complete_with_retry(request)
        except Exception as exc:
            self._record(request, None, time.monotonic() - started, f"{type(exc).__name__}: {exc}")
            raise
        if cache is not None and key is not None:
            cache.put(key, response)
        self._account(agent, response.usage)
        self._record(request, response, time.monotonic() - started, None)
        return response

    def _complete_with_retry(self, request: ChatRequest) -> ChatResponse:
        attempts = max(1, self._settings.max_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                return self._backend.complete(request)
            except LLMTransientError as exc:
                if attempt == attempts:
                    raise
                delay = self._settings.retry_backoff ** (attempt - 1)
                logger.warning(
                    "LLM call by %s failed (%s); retry %d/%d in %.1fs", request.agent, exc, attempt, attempts - 1, delay
                )
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _account(self, agent: str, usage: TokenUsage) -> None:
        with self._lock:
            self._usage = self._usage + usage
            self._usage_by_agent[agent] = self._usage_by_agent.get(agent, TokenUsage()) + usage

    def _record(self, request: ChatRequest, response: ChatResponse | None, seconds: float, error: str | None) -> None:
        if self._audit is not None:
            self._audit.record(request, response, seconds, error)
