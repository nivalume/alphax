from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from support import (
    BAR_SLUG,
    LOOK_AHEAD_STRATEGY,
    MISSING_FUNCTION_STRATEGY,
    MOMENTUM_STRATEGY,
    UNBOUNDED_STRATEGY,
    ScriptedBackend,
    SyntheticJFinance,
    bar_schema,
    event_schema,
)

from alphax_agent.coder.evaluation import EvaluationPipeline
from alphax_agent.config.settings import AlphaXSettings, WorkflowSettings
from alphax_agent.core.exceptions import ConfigurationError, EnvironmentUnavailableError, OutputParsingError
from alphax_agent.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax_agent.core.feedback import ExperimentFeedback
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.registry import scenario_registry
from alphax_agent.core.scenario import ScenarioSettings
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace, TraceRecord
from alphax_agent.env.alphax import AlphaXEnvironment, AlphaXEnvSettings
from alphax_agent.env.local import LocalEnvironment
from alphax_agent.llm.client import LLMClient
from alphax_agent.scenarios.crypto_strategy import CryptoStrategyScenario, CryptoStrategySettings
from alphax_agent.scenarios.crypto_strategy.agents import (
    SELECTED,
    ExperimentDraft,
    StrategyCoderAgent,
    StrategyDesignAgent,
    StrategyDesignPolicy,
    StrategyDraft,
    StrategyFeedbackAgent,
    StrategyHypothesisAgent,
)
from alphax_agent.scenarios.crypto_strategy.data import (
    NO_DATA_SCHEMA,
    StrategyDataCatalog,
    StrategyDataExporter,
    render_data_schemas,
)
from alphax_agent.scenarios.crypto_strategy.evaluation import (
    StrategyExecutionStage,
    StrategyFormatStage,
    StrategyLookAheadStage,
)
from alphax_agent.scenarios.crypto_strategy.prompts import (
    StrategyAnalystPrompt,
    StrategyDesignerPrompt,
    StrategyProposerPrompt,
)
from alphax_agent.scenarios.crypto_strategy.runner import AlphaXStrategyRunner, flatten
from alphax_agent.scenarios.crypto_strategy.workspace import StrategyWorkspace, StrategyWorkspaceFactory
from alphax_agent.scenarios.qlib_factor import QlibFactorScenario, QlibFactorSettings
from alphax_agent.workflow.artifacts import ArtifactStore
from alphax_agent.workflow.graph import ResearchWorkflow

PROMPTS = PromptLibrary.from_package("alphax_agent.scenarios.crypto_strategy")
PROPOSER_PROMPT = StrategyProposerPrompt(library=PROMPTS, data_schemas="bars and funding")
DESIGNER_PROMPT = StrategyDesignerPrompt(library=PROMPTS, data_schemas="bars and funding")

HYPOTHESIS = {
    "statement": "Forced liquidations push price past flow equilibrium and it partly gives the move back.",
    "rationale": "Levered longs are liquidated at any price; the transfer persists because liquidity vanishes.",
    "observation": "",
    "justification": "Liquidation cascades exist in any period.",
    "knowledge": "",
}


def strategy(name: str = "CascadeReversion", description: str = "[Mean Reversion] Buys the displacement back.") -> dict:
    return {"name": name, "description": description}


def record(loop: int, *, names: tuple[str, ...] = (), next_hypothesis: str = "") -> TraceRecord:
    experiment = Experiment(
        hypothesis=Hypothesis(**HYPOTHESIS), tasks=[Task(name=name, description="a strategy") for name in names]
    )
    feedback = ExperimentFeedback(decision=False, reason="not run", next_hypothesis=next_hypothesis)
    return TraceRecord(loop=loop, hypothesis=experiment.hypothesis, experiment=experiment, feedback=feedback)


def write_schema(root: Path, payload: dict) -> None:
    """A schema file under ``<root>/crypto/schemas``, as the jfinance pipeline writes it."""
    folder = root / "crypto" / "schemas"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{payload['slug']}.v{payload['version']}.json").write_text(json.dumps(payload))


def write_stream(root: Path, slug: str, version: int = 1, symbol: str = "BTCUSDT-PERP.BINANCE") -> None:
    (root / "crypto" / f"{slug}.v{version}" / f"{symbol}-1-DAY-LAST-EXTERNAL").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A jfinance store: a bar dataset with a stream, and a funding dataset (an event dataset)."""
    root = tmp_path / "jfinance"
    write_schema(root, bar_schema(annotations={"close": {"description": "Closing price of the bar"}}))
    write_stream(root, BAR_SLUG)
    write_schema(root, event_schema())
    return root


@pytest.fixture(scope="module")
def store() -> SyntheticJFinance:
    return SyntheticJFinance()


@pytest.fixture
def catalog_dir(tmp_path: Path, store: SyntheticJFinance) -> Path:
    return store.write_catalog(tmp_path / "cache")


def scenario_options(data_root: Path, cache: Path, **extra: object) -> dict:
    return {"alphax": {"data_dir": str(data_root)}, "cache_dir": str(cache), **extra}


class TestSettings:
    def test_both_scenarios_extend_the_framework_settings(self) -> None:
        assert issubclass(CryptoStrategySettings, ScenarioSettings)
        assert issubclass(QlibFactorSettings, ScenarioSettings)
        assert CryptoStrategySettings().environment == "alphax" and QlibFactorSettings().environment == "qlib"
        # The framework holds what every scenario shares; the period is the scenario's own.
        assert set(ScenarioSettings.model_fields) == {"environment"}

    @pytest.mark.parametrize(
        ("scenario", "environment"), [(CryptoStrategyScenario, "qlib"), (QlibFactorScenario, "alphax")]
    )
    def test_an_unsupported_environment_is_refused(self, scenario: type, environment: str) -> None:
        with pytest.raises(ConfigurationError, match=f"cannot run in the '{environment}' environment"):
            scenario.from_options({"environment": environment})

    def test_an_unknown_environment_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="crypto_strategy"):
            CryptoStrategyScenario.from_options({"environment": "docker"})

    @pytest.mark.parametrize("interval", ["1d", "day", "Daily", "1DAY"])
    def test_daily_spellings_mean_1d(self, interval: str) -> None:
        assert AlphaXEnvSettings(interval=interval).interval == "1d"

    @pytest.mark.parametrize("interval", ["1h", "5m", "1w"])
    def test_other_intervals_are_refused(self, interval: str) -> None:
        with pytest.raises(ValueError, match="only daily bars"):
            AlphaXEnvSettings(interval=interval)

    def test_the_period_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="start < oos_start < end"):
            CryptoStrategySettings.model_validate({"period": {"start": "2024-01-01", "oos_start": "2023-01-01"}})
        period = CryptoStrategySettings.model_validate(
            {"period": {"start": "2021-01-01", "oos_start": "2023-01-01", "end": "2024-01-01"}}
        ).period
        assert [str(day) for day in period.out_of_sample] == ["2023-01-01", "2024-01-01"]

    def test_paths_are_expanded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("JF_HOME", str(tmp_path))
        settings = CryptoStrategySettings.model_validate(
            {"alphax": {"data_dir": "$JF_HOME/store"}, "cache_dir": "$JF_HOME/cache"}
        )
        assert settings.alphax.data_path == (tmp_path / "store").resolve()
        assert settings.alphax.schema_path == (tmp_path / "store" / "crypto" / "schemas").resolve()
        assert settings.cache_path == (tmp_path / "cache").resolve()
        assert not str(CryptoStrategySettings().alphax.data_path).startswith("~")

    def test_the_interpreter_path_keeps_the_venv_link(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        venv = tmp_path / "venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").symlink_to(sys.executable)
        monkeypatch.chdir(tmp_path)
        settings = AlphaXEnvSettings(python_bin="venv/bin/python")
        assert settings.python_bin == str(venv / "python")
        assert AlphaXEnvSettings(python_bin="python3").python_bin == "python3"


class TestDataSchemas:
    def settings(self, root: Path, *schemas: str) -> AlphaXEnvSettings:
        return AlphaXEnvSettings(data_dir=root, schemas=list(schemas or [BAR_SLUG]))

    def test_renders_a_bar_dataset_with_its_values(self, data_root: Path) -> None:
        text = render_data_schemas(self.settings(data_root))
        assert f"### `{BAR_SLUG}` (SyntheticBar, 1d bars)" in text and "Synthetic daily bars." in text
        assert "| `close` | decimal(38,18) | Closing price of the bar |" in text
        assert "| `trade_count` | int64 |  |" in text
        assert "ts_open" not in text and "bar_type" not in text

    def test_leaves_out_a_dataset_without_rows_or_without_bars(self, data_root: Path) -> None:
        write_schema(data_root, bar_schema("binance.vision.market.crypto.mark.bar.raw"))
        settings = self.settings(
            data_root, BAR_SLUG, "binance.vision.market.crypto.mark.bar.raw", "binance.vision.market.crypto.funding.raw"
        )
        text = render_data_schemas(settings)
        assert BAR_SLUG in text and "mark.bar.raw" not in text and "funding.raw" not in text

    def test_keeps_the_latest_version_of_a_schema(self, data_root: Path) -> None:
        write_schema(data_root, bar_schema(version=2, extra=("close_v2",)))
        write_stream(data_root, BAR_SLUG, version=2)
        assert "close_v2" in render_data_schemas(self.settings(data_root))

    def test_a_broken_schema_file_is_skipped(self, data_root: Path) -> None:
        (data_root / "crypto" / "schemas" / "broken.v1.json").write_text("{not json")
        assert BAR_SLUG in render_data_schemas(self.settings(data_root))

    def test_no_schemas_renders_nothing(self, tmp_path: Path) -> None:
        assert render_data_schemas(self.settings(tmp_path)) == ""

    def test_the_export_names_the_unusable_datasets_and_the_usable_ones(self, data_root: Path, tmp_path: Path) -> None:
        settings = CryptoStrategySettings.model_validate(
            {"alphax": {"data_dir": str(data_root), "schemas": ["binance.vision.market.crypto.funding.raw"]}}
        )
        exporter = StrategyDataExporter(AlphaXEnvironment(settings.alphax), settings, StrategyDataCatalog(tmp_path))
        with pytest.raises(EnvironmentUnavailableError, match="not a bar dataset") as error:
            exporter.config()
        assert f"rows on disk: {BAR_SLUG}" in str(error.value)


class TestCatalog:
    def test_describes_the_panel_to_the_coder(self, catalog_dir: Path) -> None:
        text = StrategyDataCatalog(catalog_dir).describe()
        assert "dict of NumPy float64 arrays of shape (bars, instruments)" in text
        assert "`amount` is the `quote_volume` column" in text
        assert "3 instruments over 300 bars" in text and "6 instruments over 1461 bars" in text

    def test_profiles_a_field_of_one_instrument(self, catalog_dir: Path) -> None:
        catalog = StrategyDataCatalog(catalog_dir)
        text = catalog.profile("close", "SYN0USDT-PERP.BINANCE")
        assert "instruments=1" in text and "close |" in text and "volume |" not in text
        assert "does not exist" in catalog.profile("vwap")


@pytest.fixture
def local_env() -> LocalEnvironment:
    """The harness's positions mode needs numpy only, so this interpreter runs it."""
    return LocalEnvironment(default_timeout=120)


class TestStages:
    @pytest.fixture
    def pipeline(self, catalog_dir: Path, local_env: LocalEnvironment) -> EvaluationPipeline:
        catalog = StrategyDataCatalog(catalog_dir)
        return EvaluationPipeline(
            [
                StrategyExecutionStage(local_env, 120),
                StrategyFormatStage(catalog, 0.01),
                StrategyLookAheadStage(local_env, catalog, 120),
            ]
        )

    def evaluate(self, pipeline: EvaluationPipeline, catalog_dir: Path, tmp_path: Path, code: str):
        task = Task(name="Rule", description="a rule")
        workspace = StrategyWorkspaceFactory(StrategyDataCatalog(catalog_dir)).create(task, tmp_path / "ws")
        workspace.files[StrategyWorkspace.ENTRY] = code
        return pipeline.evaluate(task, workspace), workspace

    @staticmethod
    def stage(feedback, name: str):
        return next(stage for stage in feedback.stages if stage.stage == name)

    def test_a_causal_bounded_strategy_is_accepted(
        self, pipeline: EvaluationPipeline, catalog_dir: Path, tmp_path: Path
    ) -> None:
        feedback, workspace = self.evaluate(pipeline, catalog_dir, tmp_path, MOMENTUM_STRATEGY)
        assert feedback.accepted, feedback.render()
        assert "params {'lookback': 20}" in self.stage(feedback, "execution").message
        assert (workspace.path / StrategyWorkspace.HARNESS).exists()
        assert StrategyWorkspace.HARNESS not in workspace.files

    def test_a_strategy_that_reads_the_next_bar_is_rejected(
        self, pipeline: EvaluationPipeline, catalog_dir: Path, tmp_path: Path
    ) -> None:
        feedback, _ = self.evaluate(pipeline, catalog_dir, tmp_path, LOOK_AHEAD_STRATEGY)
        assert not feedback.accepted
        assert "change when the bars after it are removed" in self.stage(feedback, "look_ahead").message

    def test_positions_outside_the_bounds_are_rejected(
        self, pipeline: EvaluationPipeline, catalog_dir: Path, tmp_path: Path
    ) -> None:
        feedback, _ = self.evaluate(pipeline, catalog_dir, tmp_path, UNBOUNDED_STRATEGY)
        format_stage = self.stage(feedback, "format")
        assert not format_stage.passed and "FAIL bounded" in format_stage.message

    def test_a_file_without_positions_fails_with_the_reason(
        self, pipeline: EvaluationPipeline, catalog_dir: Path, tmp_path: Path
    ) -> None:
        feedback, _ = self.evaluate(pipeline, catalog_dir, tmp_path, MISSING_FUNCTION_STRATEGY)
        execution = self.stage(feedback, "execution")
        assert not execution.passed and "must define a function positions(data, **params)" in execution.message

    def test_the_agent_cannot_replace_the_harness(
        self, pipeline: EvaluationPipeline, catalog_dir: Path, tmp_path: Path
    ) -> None:
        task = Task(name="Rule", description="a rule")
        workspace = StrategyWorkspaceFactory(StrategyDataCatalog(catalog_dir)).create(task, tmp_path / "ws")
        workspace.files.update(
            {StrategyWorkspace.ENTRY: MOMENTUM_STRATEGY, StrategyWorkspace.HARNESS: "raise SystemExit"}
        )
        assert pipeline.evaluate(task, workspace).accepted

    @pytest.mark.parametrize(
        ("positions", "failed"),
        [
            (np.zeros((10, 2)), "active"),
            (np.full((10, 2), 0.5), "varying"),
            (np.full((10, 3), 0.5), "shape"),
            (np.where(np.eye(10, 2) > 0, np.inf, 0.5), "finite"),
        ],
    )
    def test_format_checks(self, catalog_dir: Path, positions: np.ndarray, failed: str) -> None:
        close = np.ones((10, 2))
        checks = {
            name: ok
            for name, ok, _ in StrategyFormatStage(StrategyDataCatalog(catalog_dir), 0.01).checks(positions, close)
        }
        assert checks[failed] is False


class TestRunner:
    def test_flattens_the_metrics_of_a_segment_and_drops_missing_values(self) -> None:
        metrics = {"portfolio.sharpe": 1.5, "portfolio.total_return": None, "trade_count": 3, "flag": True}
        assert flatten("is", metrics) == {"is.portfolio.sharpe": 1.5, "is.trade_count": 3.0}

    def test_the_backtest_period_is_in_nanoseconds(self, catalog_dir: Path, tmp_path: Path) -> None:
        settings = CryptoStrategySettings.model_validate(
            {"period": {"start": "2021-01-01", "oos_start": "2023-01-01", "end": "2024-01-01"}}
        )
        config = AlphaXStrategyRunner(
            AlphaXEnvironment(settings.alphax), settings, StrategyDataCatalog(catalog_dir), tmp_path
        ).backtest_config()
        assert config["oos_start_ns"] == int(np.datetime64("2023-01-01", "ns").astype(np.int64))
        assert config["ann_factor"] == 365.0 and config["signal_delay"] == 1


class TestFeedbackAgent:
    def test_shows_in_sample_metrics_and_lets_the_rule_decide(self, llm: LLMClient, backend: ScriptedBackend) -> None:
        backend.on(
            "strategy_analyst",
            [
                json.dumps(
                    {
                        "observations": "Sharpe rose.",
                        "hypothesis_evaluation": "Supported.",
                        "hypothesis_supported": True,
                        "next_hypothesis": "Gate on volatility.",
                        "reason": "Better book.",
                    }
                )
            ],
        )
        settings = CryptoStrategySettings()
        agent = StrategyFeedbackAgent(
            llm, StrategyAnalystPrompt(library=PROMPTS, data_schemas="bars", settings=settings), settings.promotion
        )
        experiment = Experiment(
            hypothesis=Hypothesis(**HYPOTHESIS),
            tasks=[Task(name="A", description="a rule")],
            result=ExperimentResult(
                metrics={"is.portfolio.sharpe": 1.1, "is.portfolio.max_drawdown": -0.2}, artifacts={SELECTED: "A"}
            ),
        )
        feedback = agent.summarize(experiment, Trace())
        assert feedback.decision and feedback.next_hypothesis == "Gate on volatility."
        system, user = (message.content for message in backend.requests_of("strategy_analyst")[0].messages)
        assert "held out and never reported to you" in system and "is.portfolio.sharpe rises" in system
        assert "backtest of: A" in user and "| is.portfolio.sharpe | n/a | 1.1000 |" in user


class TestScenario:
    @pytest.fixture
    def scenario(self, data_root: Path, catalog_dir: Path) -> CryptoStrategyScenario:
        return CryptoStrategyScenario.from_options(scenario_options(data_root, catalog_dir))

    def test_is_registered(self) -> None:
        assert scenario_registry.get("crypto_strategy") is CryptoStrategyScenario
        assert "crypto_strategy" in scenario_registry.names()
        assert CryptoStrategyScenario.supported_environments == ("alphax",)

    @staticmethod
    def proposer_system(
        scenario: CryptoStrategyScenario, llm: LLMClient, backend: ScriptedBackend, tmp_path: Path
    ) -> str:
        """The system prompt of the proposer of a freshly assembled run."""
        backend.on("strategy_proposer", lambda request: json.dumps(HYPOTHESIS))
        parts = scenario.assemble(llm, AlphaXSettings(), tmp_path)
        parts.proposer.propose(Trace())
        return backend.requests_of("strategy_proposer")[-1].messages[0].content

    def test_the_proposal_agents_are_told_the_market_and_the_data(
        self, scenario: CryptoStrategyScenario, llm: LLMClient, backend: ScriptedBackend, tmp_path: Path
    ) -> None:
        parts = scenario.assemble(llm, AlphaXSettings(), tmp_path)
        for prompt in (parts.proposer.prompt, parts.designer.prompt):
            sections = prompt.sections()
            assert set(sections) == {"background", "data"}
            assert "linear perpetual futures" in sections["background"] and "The clock:" in sections["background"]
            assert BAR_SLUG in sections["data"] and "`$amount`" in sections["data"]
        system = self.proposer_system(scenario, llm, backend, tmp_path)
        assert "### Background" in system and "### Source data" in system and BAR_SLUG in system

    def test_every_role_is_implemented(self, scenario: CryptoStrategyScenario, llm: LLMClient, tmp_path: Path) -> None:
        parts = scenario.assemble(llm, AlphaXSettings(workflow=WorkflowSettings(history_window=4)), tmp_path)
        assert isinstance(parts.proposer, StrategyHypothesisAgent)
        assert isinstance(parts.designer, StrategyDesignAgent)
        assert isinstance(parts.runner, AlphaXStrategyRunner)
        assert isinstance(parts.summarizer, StrategyFeedbackAgent)
        assert parts.runner.establish_baseline() is None

    def test_the_coder_is_told_the_interface_the_panel_and_the_backtest(
        self, scenario: CryptoStrategyScenario, llm: LLMClient, tmp_path: Path
    ) -> None:
        parts = scenario.assemble(llm, AlphaXSettings(), tmp_path)
        coder = parts.developer._graph._coder  # type: ignore[attr-defined]
        assert isinstance(coder, StrategyCoderAgent) and coder.required_files == ("strategy.py",)
        sections = coder.prompt.sections()
        assert set(sections) == {"background", "data", "panel", "interface", "experiment_setting", "runtime"}
        assert "def positions(data, **params)" in sections["interface"] and "alphax.factors" in sections["interface"]
        assert "(bars, instruments)" in sections["panel"]
        setting = sections["experiment_setting"]
        assert "2021-01-01 to 2025-01-01" in setting and "the synthetic instruments: SYN0USDT-PERP.BINANCE" in setting
        assert "traded at the close of row t+1" in setting and "fees 0.0005" in setting

    def test_without_data_the_prompt_says_so(
        self, llm: LLMClient, backend: ScriptedBackend, tmp_path: Path, catalog_dir: Path
    ) -> None:
        scenario = CryptoStrategyScenario.from_options(scenario_options(tmp_path / "empty", catalog_dir))
        assert NO_DATA_SCHEMA in self.proposer_system(scenario, llm, backend, tmp_path)

    def test_each_assembly_reads_the_schemas_again(
        self,
        scenario: CryptoStrategyScenario,
        data_root: Path,
        llm: LLMClient,
        backend: ScriptedBackend,
        tmp_path: Path,
    ) -> None:
        assert "close_v2" not in self.proposer_system(scenario, llm, backend, tmp_path)
        write_schema(data_root, bar_schema(version=2, extra=("close_v2",)))
        write_stream(data_root, BAR_SLUG, version=2)
        assert "close_v2" in self.proposer_system(scenario, llm, backend, tmp_path)

    def test_prepare_refuses_an_interpreter_without_alphax(self, data_root: Path, catalog_dir: Path) -> None:
        options = scenario_options(data_root, catalog_dir)
        options["alphax"]["python_bin"] = sys.executable
        if os.system(f'"{sys.executable}" -c "import alphax" 2>/dev/null') == 0:
            pytest.skip("this interpreter has alphax")
        with pytest.raises(EnvironmentUnavailableError, match="uv sync"):
            CryptoStrategyScenario.from_options(options).prepare()


class TestWorkflow:
    def test_stops_after_the_design(
        self, data_root: Path, catalog_dir: Path, llm: LLMClient, backend: ScriptedBackend, tmp_path: Path
    ) -> None:
        backend.on("strategy_proposer", [json.dumps(HYPOTHESIS)])
        backend.on("strategy_designer", [json.dumps({"strategies": [strategy("A"), strategy("B")]})])
        scenario = CryptoStrategyScenario.from_options(scenario_options(data_root, catalog_dir))
        store = ArtifactStore(tmp_path / "run")
        flow = ResearchWorkflow(scenario.assemble(llm, AlphaXSettings(), tmp_path), WorkflowSettings(), store)
        trace = flow.run(until="design")
        assert len(trace) == 0 and trace.baseline is None
        assert not list(store.loops_dir.glob("*.json"))


class TestDesignPolicy:
    def policy(self, taken: tuple[str, ...] = ()) -> StrategyDesignPolicy:
        return StrategyDesignPolicy(2, taken)

    def draft(self, *names: str) -> ExperimentDraft:
        return ExperimentDraft(strategies=[StrategyDraft(name=name, description="d") for name in names])

    def test_accepts_a_design_within_the_limit(self) -> None:
        design = self.draft("A", "B")
        assert self.policy()(design) is design

    @pytest.mark.parametrize(
        ("names", "taken", "message"),
        [
            ((), (), "at least one"),
            (("A", "B", "C"), (), "at most 2"),
            (("A", "a"), (), "unique"),
            (("Old",), ("old",), "earlier loops"),
        ],
    )
    def test_rejects_bookkeeping_errors(self, names: tuple[str, ...], taken: tuple[str, ...], message: str) -> None:
        with pytest.raises(OutputParsingError, match=message):
            self.policy(taken)(self.draft(*names))

    def test_a_draft_needs_an_identifier_and_a_description(self) -> None:
        with pytest.raises(ValueError, match="identifiers"):
            StrategyDraft(name="mean reversion", description="d")
        with pytest.raises(ValueError, match="description"):
            StrategyDraft(name="A", description="  ")


class TestHypothesisAgent:
    @pytest.fixture
    def agent(self, llm: LLMClient) -> StrategyHypothesisAgent:
        return StrategyHypothesisAgent(llm, PROPOSER_PROMPT, history_window=4)

    def test_proposes_a_hypothesis_with_the_scenario_in_the_prompt(
        self, agent: StrategyHypothesisAgent, backend: ScriptedBackend
    ) -> None:
        backend.on("strategy_proposer", [json.dumps(HYPOTHESIS)])
        hypothesis = agent.propose(Trace())
        assert hypothesis.statement == HYPOTHESIS["statement"]
        system, user = (message.content for message in backend.requests_of("strategy_proposer")[0].messages)
        assert "linear perpetual futures" in system and "The clock:" in system and "bars and funding" in system
        assert "### Background" in system and "### Source data" in system and "Code interface" not in system
        assert "Mean reversion" in system and "Trend following" in system
        assert "Do not name any instrument" in system and "Falsifiable prediction" in system
        assert "JSON schema" in system
        assert "No experiment has been run yet." in user
        assert "Start with the simplest mechanism" in user and "Suggestion" not in user

    def test_the_guidance_changes_after_enough_loops(
        self, agent: StrategyHypothesisAgent, backend: ScriptedBackend
    ) -> None:
        backend.on("strategy_proposer", [json.dumps(HYPOTHESIS)])
        trace = Trace([record(loop) for loop in range(StrategyHypothesisAgent.SIMPLE_FIRST_LOOPS)])
        agent.propose(trace)
        user = backend.requests_of("strategy_proposer")[0].messages[1].content
        assert "Now vary the conditioning" in user and "Start with the simplest" not in user

    def test_shows_the_recent_history_and_the_last_suggestion(
        self, agent: StrategyHypothesisAgent, backend: ScriptedBackend
    ) -> None:
        backend.on("strategy_proposer", [json.dumps(HYPOTHESIS)])
        trace = Trace(
            [record(1, names=("OldRule",)), record(2, names=("NewerRule",), next_hypothesis="condition on funding")]
        )
        agent.propose(trace)
        user = backend.requests_of("strategy_proposer")[0].messages[1].content
        assert "Loops run so far: 2" in user
        assert "OldRule" in user and "NewerRule" in user
        assert "# Suggestion from the last analysis" in user and "condition on funding" in user


class TestDesignAgent:
    @pytest.fixture
    def agent(self, llm: LLMClient) -> StrategyDesignAgent:
        return StrategyDesignAgent(llm, DESIGNER_PROMPT, max_strategies=2, history_window=4)

    def test_designs_an_experiment_of_plain_tasks(self, agent: StrategyDesignAgent, backend: ScriptedBackend) -> None:
        backend.on("strategy_designer", [json.dumps({"strategies": [strategy("A"), strategy("B", "Longer horizon")]})])
        hypothesis = Hypothesis(**HYPOTHESIS)
        experiment = agent.design(hypothesis, Trace([record(1, names=("Earlier",))]))
        assert type(experiment) is Experiment
        assert experiment.hypothesis == hypothesis
        assert [(task.name, task.description) for task in experiment.tasks] == [
            ("A", "[Mean Reversion] Buys the displacement back."),
            ("B", "Longer horizon"),
        ]
        system, user = (message.content for message in backend.requests_of("strategy_designer")[0].messages)
        assert "between 1 and 2 strategies" in system and "Taken names: Earlier." in system
        assert "mean_reversion or trend_following" in system and "Wrap every formula in $$" in system
        assert HYPOTHESIS["statement"] in user and "Earlier" in user

    def test_asks_again_when_a_name_was_used_before(self, agent: StrategyDesignAgent, backend: ScriptedBackend) -> None:
        backend.on(
            "strategy_designer",
            [json.dumps({"strategies": [strategy("Earlier")]}), json.dumps({"strategies": [strategy("Fresh")]})],
        )
        experiment = agent.design(Hypothesis(**HYPOTHESIS), Trace([record(1, names=("Earlier",))]))
        assert [task.name for task in experiment.tasks] == ["Fresh"]
        requests = backend.requests_of("strategy_designer")
        assert len(requests) == 2 and "earlier loops" in requests[1].messages[-1].content
