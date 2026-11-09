"""The qlib factor-mining scenario."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from alphax_agent.coder.agents import ReflectionAgent
from alphax_agent.coder.costeer import CoSTEER
from alphax_agent.coder.evaluation import EvaluationPipeline, EvaluationStage
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.registry import scenario_registry
from alphax_agent.core.scenario import ResearchComponents, Scenario, ScenarioContext
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
    experiment_model = FactorExperiment
    headline_metrics = ("IC", "Rank IC", "1day.excess_return_with_cost.annualized_return")

    def __init__(self, settings: QlibFactorSettings, environment: QlibEnvironment | None = None) -> None:
        super().__init__(settings)
        self._factor_settings = settings
        self._env = environment or QlibEnvironment(settings.qlib)
        self._catalog = FactorDataCatalog(settings.data_path)
        self._prompts = PromptLibrary.from_package("alphax_agent.scenarios.qlib_factor")
        self._context: ScenarioContext | None = None

    @property
    def environment(self) -> QlibEnvironment:
        return self._env

    @property
    def catalog(self) -> FactorDataCatalog:
        return self._catalog

    def prepare(self, *, force: bool = False) -> None:
        self._env.prepare()
        FactorDataExporter(self._env, self._factor_settings, self._catalog).export(force=force)
        self._context = None

    @property
    def context(self) -> ScenarioContext:
        if self._context is None:
            settings = self._factor_settings
            render = self._prompts.render
            self._context = ScenarioContext(
                background=render("scenario.background", market=settings.market),
                data=self._catalog.describe(),
                interface=render("scenario.interface"),
                output_format=render("scenario.output_format"),
                experiment_setting=render(
                    "scenario.experiment_setting",
                    label=settings.label,
                    model_class=settings.model.get("class", "the configured model"),
                    base_count=len(settings.base_features),
                    base_names=list(settings.base_features),
                    segments=settings.segments,
                    backtest=settings.backtest,
                    benchmark=settings.benchmark,
                    promotion=settings.promotion.describe(),
                ),
                runtime=self._env.description,
            )
        return self._context

    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        settings = self._factor_settings
        context = self.context
        prompts = self._prompts
        profile = DataProfileTool(self._catalog)
        developer = CoSTEER(
            FactorCoderAgent(llm, prompts, context, tools=[profile]),
            EvaluationPipeline(self._stages(llm, context)),
            FactorWorkspaceFactory(self._catalog),
            platform.costeer,
            Path(run_dir) / "coding",
            reflector=ReflectionAgent(llm, prompts, context),
        )
        return ResearchComponents(
            proposer=FactorHypothesisAgent(
                llm, prompts, context, tools=[profile], history_window=platform.workflow.history_window
            ),
            designer=FactorDesignAgent(
                llm,
                prompts,
                context,
                max_factors=settings.max_factors_per_experiment,
                reserved_names=settings.base_features,
            ),
            developer=developer,
            runner=QlibFactorRunner(self._env, settings, self._catalog, Path(run_dir) / "backtests"),
            summarizer=FactorFeedbackAgent(llm, prompts, context, settings.promotion),
        )

    def _stages(self, llm: LLMClient, context: ScenarioContext) -> list[EvaluationStage]:
        settings = self._factor_settings
        weights = settings.stage_weights
        stages: list[EvaluationStage] = [
            FactorExecutionStage(self._env, settings.factor_timeout, weight=weights.get("execution", 1.0)),
            FactorFormatStage(self._catalog, settings.max_nan_ratio, weight=weights.get("format", 1.0)),
            LookAheadStage(self._env, self._catalog, settings.factor_timeout, weight=weights.get("look_ahead", 1.0)),
        ]
        if settings.use_critic:
            critic = FactorCriticAgent(llm, self._prompts, context)
            stages.append(FactorCriticStage(critic, weight=weights.get("critic", 1.0)))
        return stages
