"""Scenarios: the pluggable unit of the platform.

A scenario prepares a research domain and wires the five research roles for it
(:class:`ResearchComponents`). What each role's agent is told about the domain is up to that
agent's own prompt. The workflow only talks to the roles, so replacing the scenario replaces
the whole research domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ValidationError

from alphax_agent.core.exceptions import ConfigurationError
from alphax_agent.core.experiment import Experiment
from alphax_agent.core.roles import Developer, ExperimentDesigner, HypothesisProposer, Runner, Summarizer

if TYPE_CHECKING:
    from alphax_agent.config.settings import AlphaXSettings
    from alphax_agent.llm.client import LLMClient

EnvironmentName = Literal["qlib", "alphax"]
"""The execution environments the platform knows; each scenario says which of them it can run in."""


class ScenarioSettings(BaseModel):
    """Base of every scenario's options: only what means the same in every scenario.

    Everything whose meaning depends on the research domain - the data, the backtest period, the costs -
    is declared by the scenario's own subclass.
    """

    environment: EnvironmentName
    """Where the scenario's code runs; a subclass gives the default it was built for."""


@dataclass(frozen=True)
class ResearchComponents:
    proposer: HypothesisProposer
    designer: ExperimentDesigner
    developer: Developer
    runner: Runner
    summarizer: Summarizer


class Scenario(ABC):
    """Base class of research scenarios.

    Subclasses set ``name`` (registry key), ``settings_model`` (validates the scenario
    options of the run file), ``supported_environments`` (the values its ``environment``
    option may take) and ``experiment_model`` (restores saved experiments with their
    concrete task type); ``headline_metrics`` are the metrics run reports show first.
    """

    name: ClassVar[str]
    title: ClassVar[str] = ""
    settings_model: ClassVar[type[ScenarioSettings]]
    supported_environments: ClassVar[tuple[EnvironmentName, ...]]
    experiment_model: ClassVar[type[Experiment]] = Experiment
    headline_metrics: ClassVar[tuple[str, ...]] = ()

    def __init__(self, settings: ScenarioSettings) -> None:
        self.check_environment(settings)
        self._settings = settings

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        try:
            settings = cls.settings_model.model_validate(dict(options))
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid options for scenario {cls.name!r}:\n{exc}") from exc
        return cls(settings)

    @classmethod
    def check_environment(cls, settings: ScenarioSettings) -> None:
        """Refuse an environment this scenario cannot run in."""
        if settings.environment not in cls.supported_environments:
            raise ConfigurationError(
                f"Scenario {cls.name!r} cannot run in the {settings.environment!r} environment; "
                f"supported: {', '.join(cls.supported_environments)}"
            )

    @property
    def settings(self) -> ScenarioSettings:
        return self._settings

    @abstractmethod
    def prepare(self, *, force: bool = False) -> None:
        """Check the environment and build derived data; idempotent unless ``force``."""

    @abstractmethod
    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        """Create the research roles of this scenario for one run."""
