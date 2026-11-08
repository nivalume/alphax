from __future__ import annotations

import json
import warnings
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from alphax.cli import main
from alphax.config.settings import AlphaXSettings, WorkflowSettings
from alphax.core.exceptions import CoderError
from alphax.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax.core.feedback import ExperimentFeedback
from alphax.core.promotion import MetricCriterion, PromotionRule
from alphax.core.roles import Developer, ExperimentDesigner, HypothesisProposer, Runner, Summarizer
from alphax.core.scenario import ResearchComponents
from alphax.core.task import Task
from alphax.core.trace import Trace
from alphax.workflow.artifacts import ArtifactStore
from alphax.workflow.graph import CheckpointSerde, ResearchWorkflow


class CountingProposer(HypothesisProposer):
    def __init__(self) -> None:
        self.seen_history: list[int] = []

    def propose(self, trace: Trace) -> Hypothesis:
        self.seen_history.append(len(trace))
        return Hypothesis(statement=f"idea {len(trace)}", rationale="because")


class OneTaskDesigner(ExperimentDesigner):
    def design(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        return Experiment(hypothesis=hypothesis, tasks=[Task(name=f"task{len(trace)}", description="d")])


class FlakyDeveloper(Developer):
    def __init__(self, failing_loops: set[int]) -> None:
        self.failing = failing_loops

    def develop(self, experiment: Experiment, trace: Trace) -> Experiment:
        if len(trace) in self.failing:
            raise CoderError("nothing compiled")
        return experiment


class ScoreRunner(Runner):
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def establish_baseline(self) -> ExperimentResult:
        return ExperimentResult(metrics={"IC": 0.01})

    def run(self, experiment: Experiment, trace: Trace) -> Experiment:
        experiment.result = ExperimentResult(metrics={"IC": self.scores[len(trace)]})
        return experiment


class RuleSummarizer(Summarizer):
    rule = PromotionRule(primary=MetricCriterion(metric="IC"))

    def summarize(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:
        verdict = self.rule.decide(experiment.result, trace.reference_result())
        return ExperimentFeedback(decision=verdict.promote, reason=verdict.reason, next_hypothesis="more")


def components(failing: set[int], scores: list[float], proposer: CountingProposer | None = None) -> ResearchComponents:
    return ResearchComponents(
        proposer=proposer or CountingProposer(),
        designer=OneTaskDesigner(),
        developer=FlakyDeveloper(failing),
        runner=ScoreRunner(scores),
        summarizer=RuleSummarizer(),
    )


def test_loop_records_failures_and_promotions(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    proposer = CountingProposer()
    workflow = ResearchWorkflow(components({1}, [0.02, 0.0, 0.015, 0.03], proposer), WorkflowSettings(), store)
    trace = workflow.run(loops=4)

    assert [record.loop for record in trace.records] == [0, 1, 2, 3]
    assert proposer.seen_history == [0, 1, 2, 3]
    decisions = [record.feedback.decision for record in trace.records]
    assert decisions == [True, False, False, True]
    failed = trace.records[1].feedback
    assert failed.failure is not None and failed.failure.stage == "develop"
    assert failed.failure.error_type == "CoderError"
    assert trace.records[1].experiment is not None and trace.records[1].experiment.result is None
    assert trace.baseline.get("IC") == 0.01
    assert trace.sota().loop == 3
    saved = json.loads((tmp_path / "run" / "loops" / "001.json").read_text())
    assert saved["feedback"]["failure"]["stage"] == "develop"
    assert (tmp_path / "run" / "baseline.json").exists()


def test_stops_after_consecutive_failures(tmp_path: Path) -> None:
    settings = WorkflowSettings(max_consecutive_failures=2)
    workflow = ResearchWorkflow(components({0, 1, 2, 3}, [0.0] * 5), settings, ArtifactStore(tmp_path))
    trace = workflow.run(loops=5)
    assert len(trace) == 2
    assert trace.consecutive_failures() == 2


def test_resume_continues_numbering_and_keeps_best(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    ResearchWorkflow(components(set(), [0.02, 0.01]), WorkflowSettings(), store).run(loops=2)

    reopened = ArtifactStore.open(tmp_path / "run")
    history = reopened.load_records()
    proposer = CountingProposer()
    workflow = ResearchWorkflow(components(set(), [0.0, 0.0, 0.025], proposer), WorkflowSettings(), reopened)
    trace = workflow.run(loops=1, history=history, baseline=reopened.load_baseline())
    assert [record.loop for record in trace.records] == [0, 1, 2]
    assert proposer.seen_history == [2]
    assert trace.records[2].feedback.decision
    assert "0.020000 -> 0.025000" in trace.records[2].feedback.reason


def test_summary_file_and_cli_report(tmp_path: Path, capsys) -> None:
    store = ArtifactStore(tmp_path)
    trace = ResearchWorkflow(components(set(), [0.02]), WorkflowSettings(), store).run(loops=1)
    summary = json.loads(store.save_summary(trace, {}, ["IC"]).read_text())
    assert summary["accepted_loops"] == [0]
    assert summary["loops"][0]["metrics"] == {"IC": 0.02}
    assert main(["report", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "baseline  IC=0.0100" in printed and "loop   0  PROMOTED" in printed


def test_checkpointer_round_trip(tmp_path: Path) -> None:
    saver = InMemorySaver(serde=CheckpointSerde.create())
    workflow = ResearchWorkflow(
        components({1}, [0.02, 0.0]), WorkflowSettings(), ArtifactStore(tmp_path), checkpointer=saver
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        workflow.run(loops=2, thread_id="run-1")
        history = workflow.graph.get_state({"configurable": {"thread_id": "run-1"}}).values["history"]
    assert [record.loop for record in history] == [0, 1]
    assert history[1].feedback.failure.stage == "develop"


class TestSettings:
    def test_yaml_with_environment_override(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "run.yaml"
        path.write_text(
            "llm:\n  model: deepseek/deepseek-chat\n  temperature: 0.5\n"
            "costeer:\n  policy: greedy\n"
            "scenario:\n  name: qlib_factor\n  options:\n    market: csi500\n"
        )
        monkeypatch.setenv("ALPHAX_LLM__MODEL", "openai/gpt-4o")
        monkeypatch.setenv("ALPHAX_LLM__API_KEY", "secret")
        settings = AlphaXSettings.load(path)
        assert settings.llm.model == "openai/gpt-4o"
        assert settings.llm.temperature == 0.5
        assert settings.costeer.policy == "greedy"
        assert settings.scenario.options == {"market": "csi500"}
        assert settings.snapshot()["llm"]["api_key"] == "**********"

    def test_defaults_without_file(self, monkeypatch) -> None:
        monkeypatch.delenv("ALPHAX_LLM__MODEL", raising=False)
        assert AlphaXSettings.load().workflow.max_loops == 10


def test_example_config_validates() -> None:
    from alphax.core.registry import scenario_registry

    settings = AlphaXSettings.load(Path(__file__).parents[1] / "configs" / "qlib_factor.yaml")
    scenario_cls = scenario_registry.get(settings.scenario.name)
    options = scenario_cls.settings_model.model_validate(settings.scenario.options)
    assert options.promotion.guards[0].metric == "1day.excess_return_with_cost.annualized_return"
    assert settings.costeer.lesson_store == Path("~/.alphax/lessons/qlib_factor.json").expanduser()
    assert settings.llm.cache_dir is not None and "~" not in str(settings.llm.cache_dir)
