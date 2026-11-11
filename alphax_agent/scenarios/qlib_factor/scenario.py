"""The qlib factor-mining scenario."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from alphax_agent.coder.agents import ReflectionAgent
from alphax_agent.coder.costeer import CoSTEER
from alphax_agent.coder.evaluation import EvaluationPipeline, EvaluationStage
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.registry import scenario_registry
from alphax_agent.core.scenario import ResearchComponents, Scenario
from alphax_agent.env.qlib import QlibEnvironment
from alphax_agent.scenarios.qlib_factor.agents import (
    FactorCoderAgent,
    FactorCriticAgent,
    FactorDesignAgent,
    FactorFeedbackAgent,
    FactorHypothesisAgent,
)
from alphax_agent.scenarios.qlib_factor.data import FactorDataCatalog, FactorDataExporter
from alphax_agent.scenarios.qlib_factor.evaluation import (
    FactorCriticStage,
    FactorExecutionStage,
    FactorFormatStage,
    LookAheadStage,
)
from alphax_agent.scenarios.qlib_factor.prompts import (
    FactorAnalystPrompt,
    FactorCoderPrompt,
    FactorCriticPrompt,
    FactorDesignerPrompt,
    FactorProposerPrompt,
    FactorReflectorPrompt,
)
from alphax_agent.scenarios.qlib_factor.runner import QlibFactorRunner
from alphax_agent.scenarios.qlib_factor.settings import QlibFactorSettings
from alphax_agent.scenarios.qlib_factor.task import FactorExperiment
from alphax_agent.scenarios.qlib_factor.tools import DataProfileTool
from alphax_agent.scenarios.qlib_factor.workspace import FactorWorkspaceFactory

if TYPE_CHECKING:
    from alphax_agent.config.settings import AlphaXSettings
    from alphax_agent.llm.client import LLMClient


@scenario_registry.register
class QlibFactorScenario(Scenario):
    """Mine price-volume factors, implement them as Python, backtest them with a qlib model."""

    name = "qlib_factor"
    title = "Factor mining on qlib"
    settings_model = QlibFactorSettings
    supported_environments = ("qlib",)
    experiment_model = FactorExperiment
    headline_metrics = ("IC", "Rank IC", "1day.excess_return_with_cost.annualized_return")

    def __init__(self, settings: QlibFactorSettings, environment: QlibEnvironment | None = None) -> None:
        super().__init__(settings)
        self._factor_settings = settings
        self._env = environment or QlibEnvironment(settings.qlib)
        self._catalog = FactorDataCatalog(settings.data_path)
        self._prompts = PromptLibrary.from_package("alphax_agent.scenarios.qlib_factor")

    @property
    def environment(self) -> QlibEnvironment:
        return self._env

    @property
    def catalog(self) -> FactorDataCatalog:
        return self._catalog

    def prepare(self, *, force: bool = False) -> None:
        self._env.prepare()
        FactorDataExporter(self._env, self._factor_settings, self._catalog).export(force=force)

    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        settings = self._factor_settings
        library = self._prompts
        # Described once per run, after prepare() exported the data; every prompt that shows the data gets this text.
        data = self._catalog.describe()
        profile = DataProfileTool(self._catalog)
        developer = CoSTEER(
            FactorCoderAgent(
                llm,
                FactorCoderPrompt(
                    library=library, settings=settings, data_description=data, runtime=self._env.description
                ),
                tools=[profile],
            ),
            EvaluationPipeline(
                self._stages(llm, FactorCriticPrompt(library=library, settings=settings, data_description=data))
            ),
            FactorWorkspaceFactory(self._catalog),
            platform.costeer,
            Path(run_dir) / "coding",
            reflector=ReflectionAgent(llm, FactorReflectorPrompt(library=library, settings=settings)),
        )
        return ResearchComponents(
            proposer=FactorHypothesisAgent(
                llm,
                FactorProposerPrompt(library=library, settings=settings, data_description=data),
                tools=[profile],
                history_window=platform.workflow.history_window,
            ),
            designer=FactorDesignAgent(
                llm,
                FactorDesignerPrompt(library=library, settings=settings, data_description=data),
                max_factors=settings.max_factors_per_experiment,
                reserved_names=settings.base_features,
            ),
            developer=developer,
            runner=QlibFactorRunner(self._env, settings, self._catalog, Path(run_dir) / "backtests"),
            summarizer=FactorFeedbackAgent(
                llm, FactorAnalystPrompt(library=library, settings=settings), settings.promotion
            ),
        )

    def _stages(self, llm: LLMClient, critic_prompt: FactorCriticPrompt) -> list[EvaluationStage]:
        settings = self._factor_settings
        weights = settings.stage_weights
        stages: list[EvaluationStage] = [
            FactorExecutionStage(self._env, settings.factor_timeout, weight=weights.get("execution", 1.0)),
            FactorFormatStage(self._catalog, settings.max_nan_ratio, weight=weights.get("format", 1.0)),
            LookAheadStage(self._env, self._catalog, settings.factor_timeout, weight=weights.get("look_ahead", 1.0)),
        ]
        if settings.use_critic:
            critic = FactorCriticAgent(llm, critic_prompt)
            stages.append(FactorCriticStage(critic, weight=weights.get("critic", 1.0)))
        return stages
