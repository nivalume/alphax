"""Chat backends. :class:`LiteLLMBackend` reaches every provider litellm supports."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from alphax.config.settings import LLMSettings
from alphax.core.exceptions import LLMError, LLMTransientError
from alphax.llm.types import ChatRequest, ChatResponse, TokenUsage, ToolCall


class ChatBackend(ABC):
    """Sends one chat request to a model and returns its reply."""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Raise :class:`LLMTransientError` for failures worth retrying, :class:`LLMError` otherwise."""


class LiteLLMBackend(ChatBackend):
    def __init__(self, settings: LLMSettings) -> None:
        import litellm

        self._litellm = litellm
        self._settings = settings
        exceptions = litellm.exceptions
        self._transient: tuple[type[BaseException], ...] = (
            exceptions.RateLimitError,
            exceptions.APIConnectionError,
            exceptions.Timeout,
            exceptions.ServiceUnavailableError,
            exceptions.InternalServerError,
        )

    @property
    def model(self) -> str:
        return self._settings.model

    def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            raw = self._litellm.completion(**self._params(request))
        except self._transient as exc:
            raise LLMTransientError(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        return self._to_response(raw)

    @cached_property
    def supports_json_mode(self) -> bool:
        try:
            params = self._litellm.get_supported_openai_params(model=self._settings.model) or []
        except Exception:
            return False
        return "response_format" in params

    def _params(self, request: ChatRequest) -> dict[str, Any]:
        settings = self._settings
        params: dict[str, Any] = {
            "model": settings.model,
            "messages": [message.to_provider() for message in request.messages],
            "temperature": settings.temperature,
            "timeout": settings.timeout,
        }
        if settings.max_tokens is not None:
            params["max_tokens"] = settings.max_tokens
        if settings.api_base:
            params["api_base"] = settings.api_base
        if settings.api_key is not None:
            params["api_key"] = settings.api_key.get_secret_value()
        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = "auto"
        elif request.json_mode and settings.json_mode and self.supports_json_mode:
            params["response_format"] = {"type": "json_object"}
        params.update(settings.extra_params)
        return params

    def _to_response(self, raw: Any) -> ChatResponse:
        choice = raw.choices[0]
        message = choice.message
        tool_calls = [self._to_tool_call(call) for call in (getattr(message, "tool_calls", None) or [])]
        usage = getattr(raw, "usage", None)
        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                calls=1,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cost=self._cost(raw),
            ),
            model=str(getattr(raw, "model", "") or self.model),
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def _cost(self, raw: Any) -> float:
        try:
            return float(self._litellm.completion_cost(completion_response=raw) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _to_tool_call(call: Any) -> ToolCall:
        raw_arguments = call.function.arguments or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            error = None if isinstance(arguments, dict) else "arguments must be a JSON object"
        except json.JSONDecodeError as exc:
            arguments, error = {}, f"arguments are not valid JSON: {exc}"
        return ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=arguments if isinstance(arguments, dict) else {},
            argument_error=error,
        )
