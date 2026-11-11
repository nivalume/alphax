"""The prompts of the crypto strategy agents.

Every agent is told about the market and the datasets; the agents that write, review or judge code are also
told the panel interface, the backtest and the runtime. The sections are rendered here, from this scenario's
prompt file, and each agent's templates place them.
"""

from __future__ import annotations

from typing import Any

from alphax_agent.agents.prompt import AgentPrompt
from alphax_agent.coder.prompts import CoderPrompt, ReflectorPrompt
from alphax_agent.scenarios.crypto_strategy.settings import CryptoStrategySettings


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


class StrategyPrompt(CryptoPrompt):
    """The sections of the agents that see code or results: the interface and the backtest."""

    settings: CryptoStrategySettings
    universe: str = ""
    """Which instruments the backtest trades and how they were chosen, from the exported panel."""

    def _interface(self) -> str:
        """The contract of ``strategy.py``."""
        return self.library.render("scenario.interface")

    def _experiment_setting(self) -> str:
        """How a strategy is backtested and judged."""
        settings = self.settings
        return self.library.render(
            "scenario.experiment_setting",
            period=settings.period,
            backtest=settings.backtest,
            interval=settings.alphax.interval,
            ann_factor=settings.alphax.ann_factor,
            universe=self.universe or "(not exported yet)",
            promotion=settings.promotion.describe(),
            primary=settings.promotion.primary.metric,
        )


class StrategyCoderPrompt(StrategyPrompt):
    """The coder's prompt, with this scenario's ``coder.system`` and the generic ``coder.user``."""

    base_package = CoderPrompt.base_package
    system_template = CoderPrompt.system_template
    user_template = CoderPrompt.user_template
    inputs = CoderPrompt.inputs

    panel_description: str
    runtime: str
    """Interpreter and package versions of the environment the code runs in."""

    def sections(self) -> dict[str, Any]:
        return {
            "background": self._background(),
            "data": self._data(),
            "panel": self.panel_description,
            "interface": self._interface(),
            "experiment_setting": self._experiment_setting(),
            "runtime": self.runtime,
        }


class StrategyCriticPrompt(StrategyPrompt):
    system_template = "critic.system"
    user_template = "critic.user"
    inputs = ("task", "code", "position_summary")

    panel_description: str

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "panel": self.panel_description, "interface": self._interface()}


class StrategyAnalystPrompt(StrategyPrompt):
    system_template = "analyst.system"
    user_template = "analyst.user"
    inputs = ("hypothesis", "tasks", "selected", "notes", "comparison", "verdict", "promoted", "history")

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "experiment_setting": self._experiment_setting()}


class StrategyReflectorPrompt(StrategyPrompt):
    """The reflection agent's prompt, with this scenario's ``reflector.system`` and the generic ``reflector.user``."""

    base_package = ReflectorPrompt.base_package
    system_template = ReflectorPrompt.system_template
    user_template = ReflectorPrompt.user_template
    inputs = ReflectorPrompt.inputs

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "interface": self._interface()}
