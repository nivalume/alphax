"""The prompts of the two crypto strategy agents.

Both agents are told about the market and the datasets; the sections are rendered here, from this
scenario's prompt file, and each agent's templates place them.
"""

from __future__ import annotations

from typing import Any

from alphax_agent.agents.prompt import AgentPrompt


class CryptoPrompt(AgentPrompt):
    """The sections the crypto agents share."""

    data_schemas: str
    """The rendered schemas of the datasets on disk, or a statement that there are none."""

    def _background(self) -> str:
        """Who the agent is, the market and the clock."""
        return self.library.render("scenario.background")

    def _data(self) -> str:
        """How fields are referenced, and the datasets."""
        return self.library.render("scenario.data", data_schemas=self.data_schemas)


class StrategyProposerPrompt(CryptoPrompt):
    system_template = "proposer.system"
    user_template = "proposer.user"
    inputs = ("history", "loops", "simple_first", "suggestion")

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "data": self._data()}


class StrategyDesignerPrompt(CryptoPrompt):
    system_template = "designer.system"
    user_template = "designer.user"
    inputs = ("hypothesis", "history", "taken", "max_strategies")

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "data": self._data()}
