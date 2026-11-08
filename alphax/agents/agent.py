"""The agent abstraction: prompts + tools + one LLM, producing one validated output."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from alphax.agents.tool import Tool, Toolbox
from alphax.core.exceptions import AgentError, OutputParsingError
from alphax.core.prompts import PromptLibrary
from alphax.llm.client import LLMClient
from alphax.llm.parser import StructuredOutputParser
from alphax.llm.types import ChatMessage

logger = logging.getLogger(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)


class Agent(Generic[OutputT]):
    """Base class of every component that calls a language model.

    A subclass declares

    * ``name`` - used for auditing and usage accounting,
    * ``output_model`` - the pydantic model the reply must validate against,
    * ``system_template`` / ``user_template`` - keys into its :class:`PromptLibrary`,

    and exposes a typed public method (``propose``, ``implement``, ...) that builds the
    template context and calls :meth:`_ask`. The conversation loop runs the requested
    tools, feeds their output back, and asks the model to repair a reply that does not
    parse, within fixed budgets.
    """

    name: ClassVar[str]
    output_model: ClassVar[type[BaseModel]]
    system_template: ClassVar[str]
    user_template: ClassVar[str]

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptLibrary,
        tools: Sequence[Tool] = (),
        *,
        max_tool_rounds: int = 6,
        max_parse_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._prompts = self._base_prompts().merged(prompts)
        self._toolbox = Toolbox(tools)
        self._parser: StructuredOutputParser[OutputT] = StructuredOutputParser(self.output_model)  # type: ignore[arg-type]
        self._max_tool_rounds = max_tool_rounds
        self._max_parse_retries = max_parse_retries

    @property
    def tools(self) -> list[str]:
        return self._toolbox.names()

    def _ask(self, validator: Callable[[OutputT], OutputT] | None = None, /, **context: Any) -> OutputT:
        """Render the templates with ``context`` and converse until a valid output is produced.

        ``validator`` replaces :meth:`_validate` for this call, for checks that depend on
        the call's arguments.
        """
        return self._converse(self._messages(context), validator or self._validate)

    def _messages(self, context: dict[str, Any]) -> list[ChatMessage]:
        system = self._prompts.render(self.system_template, **context)
        user = self._prompts.render(self.user_template, **context)
        return [
            ChatMessage.system(f"{system}\n\n{self._parser.format_instructions()}"),
            ChatMessage.user(user),
        ]

    def _validate(self, output: OutputT) -> OutputT:
        """Domain checks on a parsed reply; raise :class:`OutputParsingError` to ask for a new one."""
        return output

    def _converse(self, messages: list[ChatMessage], validator: Callable[[OutputT], OutputT]) -> OutputT:
        tool_rounds = 0
        refused_tool_rounds = 0
        parse_failures = 0
        tools = self._toolbox.specs() if self._toolbox else None
        while True:
            response = self._llm.chat(messages, tools=tools, json_mode=tools is None, agent=self.name)
            if response.tool_calls:
                messages.append(response.as_message())
                if tool_rounds < self._max_tool_rounds:
                    tool_rounds += 1
                    messages.extend(self._toolbox.execute(call) for call in response.tool_calls)
                    continue
                refused_tool_rounds += 1
                if refused_tool_rounds > 1:
                    raise AgentError(f"{self.name} kept calling tools after its budget of {self._max_tool_rounds}")
                refusal = self._prompts.render("agent.tool_budget_exhausted")
                messages.extend(ChatMessage.tool(call, refusal) for call in response.tool_calls)
                continue
            try:
                return validator(self._parser.parse(response.content))
            except OutputParsingError as exc:
                parse_failures += 1
                logger.info("%s reply rejected (%d/%d): %s", self.name, parse_failures, self._max_parse_retries, exc)
                if parse_failures > self._max_parse_retries:
                    raise AgentError(f"{self.name} produced no valid output: {exc}") from exc
                messages.append(ChatMessage.assistant(response.content))
                messages.append(ChatMessage.user(self._prompts.render("agent.repair_output", error=str(exc))))

    @staticmethod
    @cache
    def _base_prompts() -> PromptLibrary:
        return PromptLibrary.from_package("alphax.agents")
