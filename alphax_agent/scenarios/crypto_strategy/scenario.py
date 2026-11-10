"""The crypto strategy scenario: hypotheses and experiment designs for tradeable rules on perpetual futures.

Only the two proposal roles are implemented. The environment, the developer, the runner and the summarizer
are left empty (see :mod:`alphax_agent.scenarios.crypto_strategy.placeholders`), and there is no baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.registry import scenario_registry
from alphax_agent.core.scenario import ResearchComponents, Scenario, ScenarioContext
from alphax_agent.scenarios.crypto_strategy.agents import StrategyDesignAgent, StrategyHypothesisAgent
from alphax_agent.scenarios.crypto_strategy.data import NO_DATA_SCHEMA, render_data_schemas
from alphax_agent.scenarios.crypto_strategy.placeholders import PendingDeveloper, PendingRunner, PendingSummarizer
from alphax_agent.scenarios.crypto_strategy.settings import CryptoStrategySettings

if TYPE_CHECKING:
    from alphax_agent.config.settings import AlphaXSettings
    from alphax_agent.llm.client import LLMClient


@scenario_registry.register
class CryptoStrategyScenario(Scenario):
    """Propose mean-reversion and trend-following strategies for crypto perpetuals, each with its economic thesis."""

    name = "crypto_strategy"
    title = "Crypto strategy research (hypothesis and experiment design)"
    settings_model = CryptoStrategySettings

    def __init__(self, settings: CryptoStrategySettings) -> None:
        super().__init__(settings)
        self._strategy_settings = settings
        self._prompts = PromptLibrary.from_package("alphax_agent.scenarios.crypto_strategy")
        self._context: ScenarioContext | None = None

    def prepare(self, *, force: bool = False) -> None:
        # There is no environment to check and no data to derive; forget the context so that the data
        # schemas are read again the next time it is needed.
        self._context = None

    @property
    def context(self) -> ScenarioContext:
        if self._context is None:
            render = self._prompts.render
            schemas = render_data_schemas(self._strategy_settings.data_path)
            self._context = ScenarioContext(
                background=render("scenario.background"),
                data=render("scenario.data", data_schemas=schemas or NO_DATA_SCHEMA),
            )
        return self._context

    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        context = self.context
        window = platform.workflow.history_window
        return ResearchComponents(
            proposer=StrategyHypothesisAgent(llm, self._prompts, context, history_window=window),
            designer=StrategyDesignAgent(
                llm,
                self._prompts,
                context,
                max_strategies=self._strategy_settings.max_strategies_per_experiment,
                history_window=window,
            ),
            developer=PendingDeveloper(),
            runner=PendingRunner(),
            summarizer=PendingSummarizer(),
        )
