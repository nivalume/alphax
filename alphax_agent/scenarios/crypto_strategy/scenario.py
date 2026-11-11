"""The crypto strategy scenario: tradeable rules on perpetual futures, coded and backtested with alphax.

A loop proposes an economic mechanism, designs the rules that test it, implements each rule as
``strategy.py`` (target positions over a panel of bars), and backtests the accepted ones with alphax on the
jfinance data. There is no baseline: the first result is the first incumbent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from alphax_agent.coder.agents import ReflectionAgent
from alphax_agent.coder.costeer import CoSTEER
from alphax_agent.coder.evaluation import EvaluationPipeline, EvaluationStage
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.registry import scenario_registry
from alphax_agent.core.scenario import ResearchComponents, Scenario
from alphax_agent.env.alphax import AlphaXEnvironment
from alphax_agent.scenarios.crypto_strategy.agents import (
    StrategyCoderAgent,
    StrategyCriticAgent,
    StrategyDesignAgent,
    StrategyFeedbackAgent,
    StrategyHypothesisAgent,
)
from alphax_agent.scenarios.crypto_strategy.data import (
    NO_DATA_SCHEMA,
    StrategyDataCatalog,
    StrategyDataExporter,
    describe_symbols,
    render_data_schemas,
)
from alphax_agent.scenarios.crypto_strategy.evaluation import (
    StrategyCriticStage,
    StrategyExecutionStage,
    StrategyFormatStage,
    StrategyLookAheadStage,
)
from alphax_agent.scenarios.crypto_strategy.prompts import (
    StrategyAnalystPrompt,
    StrategyCoderPrompt,
    StrategyCriticPrompt,
    StrategyDesignerPrompt,
    StrategyProposerPrompt,
    StrategyReflectorPrompt,
)
from alphax_agent.scenarios.crypto_strategy.runner import AlphaXStrategyRunner
from alphax_agent.scenarios.crypto_strategy.settings import CryptoStrategySettings
from alphax_agent.scenarios.crypto_strategy.tools import PanelProfileTool
from alphax_agent.scenarios.crypto_strategy.workspace import StrategyWorkspaceFactory

if TYPE_CHECKING:
    from alphax_agent.config.settings import AlphaXSettings
    from alphax_agent.llm.client import LLMClient


@scenario_registry.register
class CryptoStrategyScenario(Scenario):
    """Propose mean-reversion and trend-following strategies for crypto perpetuals, code and backtest them."""

    name = "crypto_strategy"
    title = "Crypto strategy research on alphax"
    settings_model = CryptoStrategySettings
    supported_environments = ("alphax",)
    headline_metrics = ("is.portfolio.sharpe", "is.portfolio.annualized_return", "is.portfolio.max_drawdown")

    def __init__(self, settings: CryptoStrategySettings, environment: AlphaXEnvironment | None = None) -> None:
        super().__init__(settings)
        self._strategy_settings = settings
        self._env = environment or AlphaXEnvironment(settings.alphax)
        self._catalog = StrategyDataCatalog(settings.cache_path)
        self._prompts = PromptLibrary.from_package("alphax_agent.scenarios.crypto_strategy")

    @property
    def environment(self) -> AlphaXEnvironment:
        return self._env

    @property
    def catalog(self) -> StrategyDataCatalog:
        return self._catalog

    def prepare(self, *, force: bool = False) -> None:
        self._env.prepare()
        StrategyDataExporter(self._env, self._strategy_settings, self._catalog).export(force=force)

    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        settings = self._strategy_settings
        library = self._prompts
        window = platform.workflow.history_window
        # Read on every assembly, so that a run sees the schemas on disk when it starts.
        data_schemas = render_data_schemas(settings.alphax) or NO_DATA_SCHEMA
        # Described once per run, after prepare() exported the panels; every prompt that shows code gets it.
        panel = self._catalog.describe()
        universe = self._universe()
        shared: dict[str, Any] = {
            "library": library,
            "data_schemas": data_schemas,
            "settings": settings,
            "universe": universe,
        }
        profile = PanelProfileTool(self._catalog)
        developer = CoSTEER(
            StrategyCoderAgent(
                llm,
                StrategyCoderPrompt(**shared, panel_description=panel, runtime=self._env.description),
                tools=[profile],
            ),
            EvaluationPipeline(self._stages(llm, StrategyCriticPrompt(**shared, panel_description=panel))),
            StrategyWorkspaceFactory(self._catalog),
            platform.costeer,
            Path(run_dir) / "coding",
            reflector=ReflectionAgent(llm, StrategyReflectorPrompt(**shared)),
        )
        return ResearchComponents(
            proposer=StrategyHypothesisAgent(
                llm,
                StrategyProposerPrompt(library=library, data_schemas=data_schemas),
                history_window=window,
            ),
            designer=StrategyDesignAgent(
                llm,
                StrategyDesignerPrompt(library=library, data_schemas=data_schemas),
                max_strategies=settings.max_strategies_per_experiment,
                history_window=window,
            ),
            developer=developer,
            runner=AlphaXStrategyRunner(self._env, settings, self._catalog, Path(run_dir) / "backtests"),
            summarizer=StrategyFeedbackAgent(llm, StrategyAnalystPrompt(**shared), settings.promotion),
        )

    def _universe(self) -> str:
        universe = self._catalog.meta["universe"]
        symbols = [instrument["symbol"] for instrument in universe["instruments"]]
        return f"{universe['rule']}: {describe_symbols(symbols)}"

    def _stages(self, llm: LLMClient, critic_prompt: StrategyCriticPrompt) -> list[EvaluationStage]:
        settings = self._strategy_settings
        weights = settings.stage_weights
        timeout = settings.strategy_timeout
        stages: list[EvaluationStage] = [
            StrategyExecutionStage(self._env, timeout, weight=weights.get("execution", 1.0)),
            StrategyFormatStage(self._catalog, settings.min_active_share, weight=weights.get("format", 1.0)),
            StrategyLookAheadStage(self._env, self._catalog, timeout, weight=weights.get("look_ahead", 1.0)),
        ]
        if settings.use_critic:
            stages.append(
                StrategyCriticStage(StrategyCriticAgent(llm, critic_prompt), weight=weights.get("critic", 1.0))
            )
        return stages
