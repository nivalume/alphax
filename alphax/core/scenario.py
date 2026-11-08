"""Scenarios: the pluggable unit of the platform.

A scenario describes a research domain to the agents (:class:`ScenarioContext`) and wires
the five research roles for it (:class:`ResearchComponents`). The workflow only talks to
these two interfaces, so replacing the scenario replaces the whole research domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel

from alphax.core.experiment import Experiment
from alphax.core.roles import Developer, ExperimentDesigner, HypothesisProposer, Runner, Summarizer

if TYPE_CHECKING:
    from alphax.config.settings import AlphaXSettings
    from alphax.llm.client import LLMClient


class ScenarioContext(BaseModel):
    """What every agent is told about the research domain."""

    background: str
    data: str
    interface: str = ""
    output_format: str = ""
    experiment_setting: str = ""
    runtime: str = ""

    SECTION_TITLES: ClassVar[dict[str, str]] = {
        "background": "Background",
        "data": "Source data",
        "interface": "Code interface",
        "output_format": "Output format",
        "experiment_setting": "Experiment setting",
        "runtime": "Runtime environment",
    }

    def render(self, sections: Iterable[str] | None = None) -> str:
        names = list(sections) if sections is not None else list(self.SECTION_TITLES)
        blocks = [f"### {self.SECTION_TITLES[name]}\n{getattr(self, name)}" for name in names if getattr(self, name)]
        return "\n\n".join(blocks)


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
    options of the run file) and ``experiment_model`` (restores saved experiments with
    their concrete task type); ``headline_metrics`` are the metrics run reports show first.
    """

    name: ClassVar[str]
    title: ClassVar[str] = ""
    settings_model: ClassVar[type[BaseModel]]
    experiment_model: ClassVar[type[Experiment]] = Experiment
    headline_metrics: ClassVar[tuple[str, ...]] = ()

    def __init__(self, settings: BaseModel) -> None:
        self._settings = settings

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        return cls(cls.settings_model.model_validate(dict(options)))

    @property
    def settings(self) -> BaseModel:
        return self._settings

    @abstractmethod
    def prepare(self, *, force: bool = False) -> None:
        """Check the environment and build derived data; idempotent unless ``force``."""

    @property
    @abstractmethod
    def context(self) -> ScenarioContext: ...

    @abstractmethod
    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        """Create the research roles of this scenario for one run."""
