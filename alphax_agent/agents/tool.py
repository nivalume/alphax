"""Tools an agent may call while it works."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from alphax_agent.core.text import Excerpt
from alphax_agent.llm.types import ChatMessage, ToolCall

logger = logging.getLogger(__name__)

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class NoArguments(BaseModel):
    """Argument model of a tool that takes no input."""


class Tool(ABC, Generic[ArgsT]):
    """A function the model can call, with arguments validated by ``args_model``.

    Tool errors are returned to the model as text instead of raised, so the agent can
    correct its call; only defects in the tool itself are logged.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]] = NoArguments
    max_output_chars: ClassVar[int] = 8000

    @abstractmethod
    def run(self, args: ArgsT) -> str: ...

    def spec(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("properties", {})
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }

    def invoke(self, call: ToolCall) -> str:
        if call.argument_error is not None:
            return f"ERROR: {call.argument_error}"
        try:
            args = self.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return f"ERROR: invalid arguments for {self.name}: {exc}"
        try:
            output = self.run(args)  # type: ignore[arg-type]
        except Exception as exc:
            logger.exception("Tool %s failed", self.name)
            return f"ERROR: {self.name} failed: {type(exc).__name__}: {exc}"
        return Excerpt.middle(output, self.max_output_chars)


class Toolbox:
    """The tools of one agent, addressable by name."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ChatMessage:
        tool = self._tools.get(call.name)
        if tool is None:
            content = f"ERROR: unknown tool {call.name!r}; available tools: {json.dumps(self.names())}"
        else:
            content = tool.invoke(call)
        return ChatMessage.tool(call, content)
