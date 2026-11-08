from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from support import (
    FULL_SAMPLE_NORMALISED_FACTOR,
    GOOD_FACTOR,
    KEY_ERROR_FACTOR,
    LOOK_AHEAD_FACTOR,
    WRONG_COLUMN_FACTOR,
    ScriptedBackend,
)

from alphax.coder.evaluation import EvaluationPipeline
from alphax.core.exceptions import AgentError
from alphax.core.experiment import ExperimentResult, Hypothesis, Implementation
from alphax.core.feedback import ExperimentFeedback, ImplementationFeedback
from alphax.core.promotion import MetricCriterion, MetricGuard, PromotionRule
from alphax.core.prompts import PromptLibrary
from alphax.core.scenario import ScenarioContext
from alphax.core.trace import Trace, TraceRecord
from alphax.env.local import LocalEnvironment
from alphax.scenarios.qlib_factor.agents import FactorCriticAgent, FactorDesignAgent
from alphax.scenarios.qlib_factor.data import FactorDataCatalog
from alphax.scenarios.qlib_factor.evaluation import (
    FactorCriticStage,
    FactorExecutionStage,
    FactorFormatStage,
    LookAheadStage,
)
from alphax.scenarios.qlib_factor.library import FactorLibrary
from alphax.scenarios.qlib_factor.runner import FactorCombiner, QlibWorkflowConfig
from alphax.scenarios.qlib_factor.settings import QlibFactorSettings
from alphax.scenarios.qlib_factor.task import FactorExperiment, FactorTask
from alphax.scenarios.qlib_factor.workspace import FactorWorkspaceFactory

PROMPTS = PromptLibrary.from_package("alphax.scenarios.qlib_factor")
CONTEXT = ScenarioContext(background="factor research", data="daily bars", interface="factor.py")


@pytest.fixture
def catalog(catalog_root: Path) -> FactorDataCatalog:
    return FactorDataCatalog(catalog_root)


def factor_task(name: str = "REV5") -> FactorTask:
    return FactorTask(name=name, description="5-day reversal", formulation=r"-\frac{1}{5}\sum_{i=0}^{4} r_{t-i}")


class TestCatalog:
    def test_description_and_profile(self, catalog: FactorDataCatalog) -> None:
        text = catalog.describe()
        assert "$close" in text and "datetime" in text and "10 instruments" in text
        assert "nan_ratio" in catalog.profile("$volume")
        assert "not in the development sample" in catalog.profile(instrument="XX")
        assert catalog.cut_date > pd.Timestamp("2019-06-01")


class TestFactorEvaluation:
    @pytest.fixture
    def pipeline(self, catalog: FactorDataCatalog, backend: ScriptedBackend, llm) -> EvaluationPipeline:
        backend.on(
            "factor_critic",
            lambda request: json.dumps({"consistent": True, "score": 0.9, "critique": "Matches the specification."}),
        )
        env = LocalEnvironment(default_timeout=120)
        return EvaluationPipeline(
            [
                FactorExecutionStage(env, 120, weight=0.3),
                FactorFormatStage(catalog, max_nan_ratio=0.5, weight=0.3),
                LookAheadStage(env, catalog, 120, weight=0.2),
                FactorCriticStage(FactorCriticAgent(llm, PROMPTS, CONTEXT), weight=0.2),
            ]
        )

    def evaluate(
        self, pipeline: EvaluationPipeline, catalog: FactorDataCatalog, code: str, tmp_path: Path
    ) -> ImplementationFeedback:
        task = factor_task()
        workspace = FactorWorkspaceFactory(catalog).create(task, tmp_path / "ws")
        workspace.inject({"factor.py": code.format(name=task.name)})
        return pipeline.evaluate(task, workspace)

    def test_point_in_time_factor_is_accepted(self, pipeline, catalog, tmp_path, backend) -> None:
        result = self.evaluate(pipeline, catalog, GOOD_FACTOR, tmp_path)
        assert result.accepted, result.render()
        assert result.score == pytest.approx(0.98)
        critic_prompt = backend.requests_of("factor_critic")[0].messages[1].content
        assert "rolling(5" in critic_prompt and "shape" in critic_prompt

    @pytest.mark.parametrize("code", [LOOK_AHEAD_FACTOR, FULL_SAMPLE_NORMALISED_FACTOR])
    def test_look_ahead_is_rejected(self, pipeline, catalog, tmp_path, code, backend) -> None:
        result = self.evaluate(pipeline, catalog, code, tmp_path)
        failure = result.first_failure()
        assert failure is not None and failure.stage == "look_ahead", result.render()
        assert "change when the data after" in failure.message
        assert result.stage("critic").skipped
        assert not backend.requests_of("factor_critic")

    def test_crash_is_reported_with_the_exception(self, pipeline, catalog, tmp_path) -> None:
        result = self.evaluate(pipeline, catalog, KEY_ERROR_FACTOR, tmp_path)
        failure = result.first_failure()
        assert failure.stage == "execution" and "KeyError" in failure.message
        assert "<workspace>" in failure.message or str(tmp_path) not in failure.message

    def test_wrong_column_name_is_rejected(self, pipeline, catalog, tmp_path) -> None:
        result = self.evaluate(pipeline, catalog, WRONG_COLUMN_FACTOR, tmp_path)
        failure = result.first_failure()
        assert failure.stage == "format" and "expected exactly ['REV5']" in failure.message


class TestFactorCombiner:
    def test_drops_duplicates_and_keeps_new_information(self, catalog: FactorDataCatalog) -> None:
        frame = catalog.frame("debug")
        momentum = frame["$close"].groupby(level="instrument").pct_change(5).rename("MOM5")
        rng = np.random.default_rng(0)
        independent = pd.Series(rng.normal(size=len(frame)), index=frame.index, name="NOISE")
        candidates = pd.DataFrame({"MOM5_SCALED": momentum * 3 + 1, "NOISE": independent})
        kept, notes = FactorCombiner(0.99).deduplicate(candidates, momentum.to_frame())
        assert list(kept.columns) == ["NOISE"]
        assert "MOM5_SCALED was dropped" in notes[0]
        combined = FactorCombiner(0.99).combine([momentum.to_frame(), kept])
        assert list(combined.columns) == [("feature", "MOM5"), ("feature", "NOISE")]

    def test_duplicates_within_one_experiment(self, catalog: FactorDataCatalog) -> None:
        close = catalog.frame("debug")["$close"]
        signal = close.groupby(level="instrument").pct_change(3)
        kept, _ = FactorCombiner(0.99).deduplicate(pd.DataFrame({"A": signal, "B": -signal}), None)
        assert list(kept.columns) == ["A"]


class TestWorkflowConfig:
    def test_renders_valid_yaml(self) -> None:
        settings = QlibFactorSettings(base_features={"KLEN": "($high-$low)/$open", "Q": "Mean($close, 5)/$close"})
        config = yaml.safe_load(QlibWorkflowConfig(settings).render(with_factors=True))
        loaders = config["data_handler_config"]["data_loader"]["kwargs"]["dataloader_l"]
        assert loaders[0]["kwargs"]["config"]["feature"] == [
            ["($high-$low)/$open", "Mean($close, 5)/$close"],
            ["KLEN", "Q"],
        ]
        assert loaders[0]["kwargs"]["config"]["label"] == [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]]
        assert loaders[1]["kwargs"]["config"] == "combined_factors.parquet"
        assert config["task"]["model"]["class"] == "LGBModel"
        assert config["task"]["dataset"]["kwargs"]["segments"]["test"] == ["2017-01-01", "2020-08-01"]
        assert config["port_analysis_config"]["backtest"]["exchange_kwargs"]["limit_threshold"] == 0.095
        assert config["port_analysis_config"]["backtest"]["end_time"] == "2020-08-01"
        baseline = yaml.safe_load(QlibWorkflowConfig(settings).render(with_factors=False))
        assert len(baseline["data_handler_config"]["data_loader"]["kwargs"]["dataloader_l"]) == 1

    def test_backtest_ends_before_the_last_trading_day(self) -> None:
        settings = QlibFactorSettings()
        calendar = ["2020-07-29", "2020-07-30", "2020-07-31"]
        config = yaml.safe_load(QlibWorkflowConfig(settings, calendar).render(with_factors=False))
        assert config["port_analysis_config"]["backtest"]["end_time"] == "2020-07-30"
        assert config["task"]["dataset"]["kwargs"]["segments"]["test"][1] == "2020-08-01"


class TestPromotionRule:
    rule = PromotionRule(primary=MetricCriterion(metric="IC"), guards=[MetricGuard(metric="ret", tolerance=0.01)])

    def test_first_result_is_promoted(self) -> None:
        assert self.rule.decide(ExperimentResult(metrics={"IC": 0.01}), None).promote

    def test_needs_improvement_within_guards(self) -> None:
        reference = ExperimentResult(metrics={"IC": 0.03, "ret": 0.10})
        assert self.rule.decide(ExperimentResult(metrics={"IC": 0.04, "ret": 0.095}), reference).promote
        assert not self.rule.decide(ExperimentResult(metrics={"IC": 0.04, "ret": 0.08}), reference).promote
        assert not self.rule.decide(ExperimentResult(metrics={"IC": 0.03, "ret": 0.2}), reference).promote
        assert not self.rule.decide(ExperimentResult(metrics={"ret": 0.2}), reference).promote
        assert "IC rises by more than 0" in self.rule.describe()


class TestFactorDesign:
    def test_rejects_names_of_existing_features(self, backend: ScriptedBackend, llm) -> None:
        spec = {"description": "d", "formulation": "f", "variables": {}}
        backend.on(
            "factor_designer",
            [
                json.dumps({"factors": [{"name": "KLEN", **spec}]}),
                json.dumps({"factors": [{"name": "GAP", **spec}, {"name": "gap", **spec}]}),
                json.dumps({"factors": [{"name": "GAP_REVERSAL", **spec}]}),
            ],
        )
        agent = FactorDesignAgent(llm, PROMPTS, CONTEXT, max_factors=2, reserved_names=["KLEN"])
        experiment = agent.design(Hypothesis(statement="gaps revert", rationale="overreaction"), Trace())
        assert isinstance(experiment, FactorExperiment)
        assert [task.name for task in experiment.tasks] == ["GAP_REVERSAL"]
        retries = [request.messages[-1].content for request in backend.requests_of("factor_designer")[1:]]
        assert "already used" in retries[0] and "unique" in retries[1]

    def test_invalid_identifier_fails_after_retries(self, backend: ScriptedBackend, llm) -> None:
        bad = json.dumps({"factors": [{"name": "1bad name", "description": "d", "formulation": "f"}]})
        backend.on("factor_designer", [bad, bad, bad])
        agent = FactorDesignAgent(llm, PROMPTS, CONTEXT, max_factors=2)
        with pytest.raises(AgentError):
            agent.design(Hypothesis(statement="s", rationale="r"), Trace())


class TestFactorLibrary:
    def test_only_included_factors_of_promoted_loops(self, tmp_path: Path) -> None:
        def record(loop: int, promoted: bool, included: str) -> TraceRecord:
            task = factor_task(f"F{loop}")
            values = tmp_path / f"f{loop}.parquet"
            pd.DataFrame(
                {task.name: [1.0]},
                index=pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-02"), "A")], names=["datetime", "instrument"]),
            ).to_parquet(values)
            implementation = Implementation(
                task_id=task.id,
                workspace=FactorWorkspaceFactory(FactorDataCatalog(tmp_path)).create(task, tmp_path / str(loop)),
                feedback=ImplementationFeedback(accepted=True, score=1.0),
                artifacts={FactorLibrary.INCLUDED: included, FactorLibrary.VALUES: str(values)},
            )
            experiment = FactorExperiment(tasks=[task], implementations={task.id: implementation})
            return TraceRecord(
                loop=loop, experiment=experiment, feedback=ExperimentFeedback(decision=promoted, reason="")
            )

        library = FactorLibrary(Trace([record(0, True, "true"), record(1, False, "true"), record(2, True, "false")]))
        assert library.names() == {"F0"}
        assert list(library.load_values().columns) == ["F0"]
