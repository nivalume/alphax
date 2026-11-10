"""The prompt of one agent, as a LangChain chat prompt template."""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import BaseChatPromptTemplate
from pydantic import ConfigDict, Field, field_validator, model_validator

from alphax_agent.core.prompts import PromptLibrary


class AgentPrompt(BaseChatPromptTemplate):
    """Base class of the prompt of one agent: a system and a user message, each a template of its library.

    Every agent has its own subclass, which decides what the agent is told and how. A subclass sets

    * ``system_template`` / ``user_template`` - keys of the two templates in ``library``,
    * ``inputs`` - the variables the agent passes on each call (history, hypothesis, ...),

    declares as fields what its sections are built from (settings, a data description, ...), renders
    each section in a private method of its own and returns the ones it uses from :meth:`sections`.
    Its templates place those sections, under headings of their own choosing.

    Rendering goes through :meth:`PromptLibrary.render` rather than LangChain's jinja2 formatter, which
    has neither ``trim_blocks`` nor ``StrictUndefined``; everything else (``invoke`` checking the inputs,
    ``partial``) is LangChain's.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_template: ClassVar[str]
    user_template: ClassVar[str]
    inputs: ClassVar[tuple[str, ...]] = ()
    base_package: ClassVar[str | None] = None
    """A package whose ``prompts.yaml`` the library falls back to for the templates it does not define."""

    library: PromptLibrary
    input_variables: list[str] = Field(default_factory=list)
    """LangChain's list of required inputs; filled from ``inputs`` when not given."""

    @model_validator(mode="before")
    @classmethod
    def _declare_inputs(cls, data: Any) -> Any:
        # ``inputs`` is a class variable, not a field default, so that subclasses override it reliably.
        if isinstance(data, dict) and "input_variables" not in data:
            return {**data, "input_variables": list(cls.inputs)}
        return data

    @field_validator("library")
    @classmethod
    def _over_base(cls, library: PromptLibrary) -> PromptLibrary:
        if cls.base_package is None:
            return library
        return PromptLibrary.from_package(cls.base_package).merged(library)

    def sections(self) -> dict[str, Any]:
        """The variables this prompt fills in itself, from its own fields; none by default."""
        return {}

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        variables = {**self.sections(), **self._merge_partial_and_user_variables(**kwargs)}
        return [
            SystemMessage(content=self.library.render(self.system_template, **variables)),
            HumanMessage(content=self.library.render(self.user_template, **variables)),
        ]
