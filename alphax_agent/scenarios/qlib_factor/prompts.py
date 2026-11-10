"""The prompts of the factor agents.

:class:`FactorPrompt` renders the sections the agents share from this scenario's prompt file; each
agent's prompt picks the ones it is told about, and its templates place them.
"""

from __future__ import annotations

from typing import Any

from alphax_agent.agents.prompt import AgentPrompt
from alphax_agent.coder.prompts import CoderPrompt, ReflectorPrompt
from alphax_agent.scenarios.qlib_factor.settings import QlibFactorSettings


class FactorPrompt(AgentPrompt):
    """The sections the factor agents share."""

    settings: QlibFactorSettings

    def _background(self) -> str:
        """What the research team does, and the requirements on every factor."""
        return self.library.render("scenario.background", market=self.settings.market)

    def _interface(self) -> str:
        """How factor code reads its data and where it writes the result."""
        return self.library.render("scenario.interface")

    def _output_format(self) -> str:
        """The layout of the result file."""
        return self.library.render("scenario.output_format")

    def _experiment_setting(self) -> str:
        """How a factor is judged: label, model, segments, backtest and promotion rule."""
        settings = self.settings
        return self.library.render(
            "scenario.experiment_setting",
            label=settings.label,
            model_class=settings.model.get("class", "the configured model"),
            base_count=len(settings.base_features),
            base_names=list(settings.base_features),
            segments=settings.segments,
            backtest=settings.backtest,
            benchmark=settings.benchmark,
            promotion=settings.promotion.describe(),
        )


class FactorProposerPrompt(FactorPrompt):
    system_template = "proposer.system"
    user_template = "proposer.user"
    inputs = ("history", "library", "best", "loops", "suggestion", "tools")

    data_description: str

    def sections(self) -> dict[str, Any]:
        return {
            "background": self._background(),
            "data": self.data_description,
            "experiment_setting": self._experiment_setting(),
        }


class FactorDesignerPrompt(FactorPrompt):
    system_template = "designer.system"
    user_template = "designer.user"
    inputs = ("hypothesis", "library", "reserved", "max_factors")

    data_description: str

    def sections(self) -> dict[str, Any]:
        return {
            "background": self._background(),
            "data": self.data_description,
            "experiment_setting": self._experiment_setting(),
        }


class FactorCriticPrompt(FactorPrompt):
    system_template = "critic.system"
    user_template = "critic.user"
    inputs = ("task", "code", "value_summary")

    data_description: str

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "data": self.data_description, "interface": self._interface()}


class FactorAnalystPrompt(FactorPrompt):
    system_template = "analyst.system"
    user_template = "analyst.user"
    inputs = ("hypothesis", "tasks", "included", "notes", "comparison", "verdict", "promoted", "history")

    def sections(self) -> dict[str, Any]:
        return {"background": self._background(), "experiment_setting": self._experiment_setting()}


class FactorCoderPrompt(FactorPrompt):
    """The coder's prompt, with this scenario's ``coder.system`` and the generic ``coder.user``."""

    base_package = CoderPrompt.base_package
    system_template = CoderPrompt.system_template
    user_template = CoderPrompt.user_template
    inputs = CoderPrompt.inputs

    data_description: str
    runtime: str
    """Interpreter and package versions of the environment the code runs in."""

    def sections(self) -> dict[str, Any]:
        return {
            "background": self._background(),
            "data": self.data_description,
            "interface": self._interface(),
            "output_format": self._output_format(),
            "experiment_setting": self._experiment_setting(),
            "runtime": self.runtime,
        }


class FactorReflectorPrompt(FactorPrompt):
    """The reflection agent's prompt, with this scenario's ``reflector.system`` and the generic ``reflector.user``."""

    base_package = ReflectorPrompt.base_package
    system_template = ReflectorPrompt.system_template
    user_template = ReflectorPrompt.user_template
    inputs = ReflectorPrompt.inputs

    def sections(self) -> dict[str, Any]:
        return {
            "background": self._background(),
            "interface": self._interface(),
            "output_format": self._output_format(),
        }
