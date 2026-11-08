"""Provider-neutral chat types."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    argument_error: str | None = Field(default=None, description="Set when the model sent arguments that are not JSON")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall] | None = None) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, call: ToolCall, content: str) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=call.id, name=call.name)

    def to_provider(self) -> dict[str, Any]:
        """OpenAI-compatible message dict, the format litellm accepts for every provider."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None and self.role == "tool":
            message["name"] = self.name
        return message


class TokenUsage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost=self.cost + other.cost,
        )


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    json_mode: bool = False
    agent: str = "anonymous"

    def cache_key(self, model: str, temperature: float) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [message.to_provider() for message in self.messages],
            "tools": self.tools,
            "json_mode": self.json_mode,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ChatResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str | None = None
    cached: bool = False

    def as_message(self) -> ChatMessage:
        return ChatMessage.assistant(self.content, self.tool_calls)
