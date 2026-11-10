from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import ScriptedBackend

from alphax_agent.config.settings import AlphaXSettings, WorkflowSettings
from alphax_agent.core.exceptions import ConfigurationError, OutputParsingError
from alphax_agent.core.experiment import Experiment, Hypothesis
from alphax_agent.core.feedback import ExperimentFeedback
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.registry import scenario_registry
from alphax_agent.core.scenario import ScenarioContext
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace, TraceRecord
from alphax_agent.llm.client import LLMClient
from alphax_agent.scenarios.crypto_strategy import CryptoStrategyScenario, CryptoStrategySettings
from alphax_agent.scenarios.crypto_strategy.agents import (
    ExperimentDraft,
    StrategyDesignAgent,
    StrategyDesignPolicy,
    StrategyDraft,
    StrategyHypothesisAgent,
)
from alphax_agent.scenarios.crypto_strategy.data import NO_DATA_SCHEMA, render_data_schemas
from alphax_agent.workflow.artifacts import ArtifactStore
from alphax_agent.workflow.graph import ResearchWorkflow

PROMPTS = PromptLibrary.from_package("alphax_agent.scenarios.crypto_strategy")
CONTEXT = ScenarioContext(background="a crypto desk", data="bars and funding")

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


def write_schema(root: Path, slug: str, columns: list[dict], annotations: dict | None = None, version: int = 1) -> None:
    (root / "schemas").mkdir(parents=True, exist_ok=True)
    payload = {"slug": slug, "version": version, "name": slug.split(".")[-1], "columns": columns}
    if annotations:
        payload["annotations"] = {"fields": annotations}
    (root / "schemas" / f"{slug}.v{version}.json").write_text(json.dumps(payload))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_schema(
        root,
        "market.crypto.bar",
        [{"name": "ts_open", "type": "int64"}, {"name": "close", "type": "decimal", "precision": 18, "scale": 8}],
        {"close": {"description": "Closing price of the bar"}},
    )
    (root / "crypto/data/crypto_bar").mkdir(parents=True)
    # The funding schema exists, its rows are only in a DuckDB file.
    write_schema(root, "market.crypto_funding_rate", [{"name": "funding_rate", "type": "float64"}])
    (root / "crypto/crypto_funding_rate.duckdb").touch()
    return root


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


class TestDataSchemas:
    def test_renders_the_datasets_whose_rows_are_on_disk(self, data_root: Path) -> None:
        text = render_data_schemas(data_root)
        assert "### `market.crypto.bar` (bar, bar)" in text
        assert "| `close` | decimal(18,8) | Closing price of the bar |" in text
        assert "### `market.crypto_funding_rate`" in text and "| `funding_rate` | float64 |" in text

    def test_skips_a_schema_whose_rows_are_missing(self, data_root: Path) -> None:
        (data_root / "crypto/crypto_funding_rate.duckdb").unlink()
        text = render_data_schemas(data_root)
        assert "market.crypto.bar" in text and "funding" not in text

    def test_keeps_the_latest_version_of_a_schema(self, data_root: Path) -> None:
        write_schema(data_root, "market.crypto.bar", [{"name": "close_v2", "type": "float64"}], version=2)
        text = render_data_schemas(data_root)
        assert "close_v2" in text and "ts_open" not in text

    def test_a_broken_schema_file_is_skipped(self, data_root: Path) -> None:
        (data_root / "schemas" / "broken.v1.json").write_text("{not json")
        assert "market.crypto.bar" in render_data_schemas(data_root)

    def test_no_schemas_renders_nothing(self, tmp_path: Path) -> None:
        assert render_data_schemas(tmp_path) == ""


class TestHypothesisAgent:
    @pytest.fixture
    def agent(self, llm: LLMClient) -> StrategyHypothesisAgent:
        return StrategyHypothesisAgent(llm, PROMPTS, CONTEXT, history_window=4)

    def test_proposes_a_hypothesis_with_the_scenario_in_the_prompt(
        self, agent: StrategyHypothesisAgent, backend: ScriptedBackend
    ) -> None:
        backend.on("strategy_proposer", [json.dumps(HYPOTHESIS)])
        hypothesis = agent.propose(Trace())
        assert hypothesis.statement == HYPOTHESIS["statement"]
        system, user = (message.content for message in backend.requests_of("strategy_proposer")[0].messages)
        assert "a crypto desk" in system and "bars and funding" in system
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
        return StrategyDesignAgent(llm, PROMPTS, CONTEXT, max_strategies=2, history_window=4)

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


class TestScenario:
    @pytest.fixture
    def scenario(self, data_root: Path) -> CryptoStrategyScenario:
        return CryptoStrategyScenario.from_options({"data_root_dir": str(data_root)})

    def test_is_registered(self) -> None:
        assert scenario_registry.get("crypto_strategy") is CryptoStrategyScenario
        assert "crypto_strategy" in scenario_registry.names()

    def test_settings_expand_the_data_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("JARVIS_HOME", str(tmp_path))
        settings = CryptoStrategySettings(data_root_dir=Path("$JARVIS_HOME/data"))
        assert settings.data_path == (tmp_path / "data").resolve()
        assert not str(CryptoStrategySettings().data_path).startswith("~")

    def test_context_describes_the_market_and_the_data_only(self, scenario: CryptoStrategyScenario) -> None:
        context = scenario.context
        assert "linear perpetual futures" in context.background and "The clock:" in context.background
        assert "market.crypto.bar" in context.data
        assert not (context.interface or context.output_format or context.experiment_setting or context.runtime)
        rendered = context.render()
        assert "### Background" in rendered and "### Source data" in rendered and "Code interface" not in rendered

    def test_context_without_data_says_so(self, tmp_path: Path) -> None:
        scenario = CryptoStrategyScenario(CryptoStrategySettings(data_root_dir=tmp_path))
        assert NO_DATA_SCHEMA in scenario.context.data

    def test_prepare_rereads_the_schemas(self, scenario: CryptoStrategyScenario, data_root: Path) -> None:
        assert "close_v2" not in scenario.context.data
        write_schema(data_root, "market.crypto.bar", [{"name": "close_v2", "type": "float64"}], version=2)
        scenario.prepare()
        assert "close_v2" in scenario.context.data

    def test_only_the_proposal_roles_are_implemented(
        self, scenario: CryptoStrategyScenario, llm: LLMClient, tmp_path: Path
    ) -> None:
        parts = scenario.assemble(llm, AlphaXSettings(workflow=WorkflowSettings(history_window=4)), tmp_path)
        assert isinstance(parts.proposer, StrategyHypothesisAgent)
        assert isinstance(parts.designer, StrategyDesignAgent)
        assert parts.runner.establish_baseline() is None
        experiment = Experiment()
        for call in (
            lambda: parts.developer.develop(experiment, Trace()),
            lambda: parts.runner.run(experiment, Trace()),
            lambda: parts.summarizer.summarize(experiment, Trace()),
        ):
            with pytest.raises(ConfigurationError, match="--until design"):
                call()


class TestWorkflow:
    @pytest.fixture
    def workflow(
        self, data_root: Path, llm: LLMClient, backend: ScriptedBackend, tmp_path: Path
    ) -> tuple[ResearchWorkflow, ArtifactStore]:
        backend.on("strategy_proposer", [json.dumps(HYPOTHESIS)])
        backend.on("strategy_designer", [json.dumps({"strategies": [strategy("A"), strategy("B")]})])
        scenario = CryptoStrategyScenario.from_options({"data_root_dir": str(data_root)})
        store = ArtifactStore(tmp_path / "run")
        parts = scenario.assemble(llm, AlphaXSettings(), tmp_path)
        return ResearchWorkflow(parts, WorkflowSettings(), store), store

    def test_stops_after_the_design(self, workflow: tuple[ResearchWorkflow, ArtifactStore]) -> None:
        flow, store = workflow
        trace = flow.run(until="design")
        assert len(trace) == 0 and trace.baseline is None
        assert not list(store.loops_dir.glob("*.json"))

    def test_a_full_loop_is_recorded_as_failed_at_the_missing_developer(
        self, workflow: tuple[ResearchWorkflow, ArtifactStore]
    ) -> None:
        flow, store = workflow
        trace = flow.run(loops=1)
        assert trace.baseline is None
        (saved,) = trace.records
        assert saved.hypothesis is not None and saved.hypothesis.statement == HYPOTHESIS["statement"]
        assert saved.experiment is not None and [task.name for task in saved.experiment.tasks] == ["A", "B"]
        assert saved.feedback.failure is not None and saved.feedback.failure.stage == "develop"
        (restored,) = store.load_records(CryptoStrategyScenario.experiment_model)
        assert restored.experiment is not None and [task.name for task in restored.experiment.tasks] == ["A", "B"]
