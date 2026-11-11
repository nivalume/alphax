"""What a run leaves on disk for a viewer: reasoning, call context, step events and the run record."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from support import ScriptedBackend, SyntheticJFinance

from alphax_agent.agents.agent import Agent
from alphax_agent.agents.prompt import AgentPrompt
from alphax_agent.app import AlphaXApplication
from alphax_agent.config.settings import AlphaXSettings, LLMSettings, WorkflowSettings
from alphax_agent.core.exceptions import CoderError
from alphax_agent.core.experiment import Experiment, Hypothesis
from alphax_agent.core.feedback import ExperimentFeedback
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.roles import Developer, ExperimentDesigner, HypothesisProposer, Runner, Summarizer
from alphax_agent.core.scenario import ResearchComponents
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace
from alphax_agent.llm.backend import ChatBackend, LiteLLMBackend
from alphax_agent.llm.cache import ResponseCache
from alphax_agent.llm.client import LLMAuditLog, LLMClient
from alphax_agent.llm.context import call_context, current_context
from alphax_agent.llm.types import ChatMessage, ChatRequest, ChatResponse, TokenUsage
from alphax_agent.scenarios.crypto_strategy import CryptoStrategyScenario, CryptoStrategySettings
from alphax_agent.workflow.artifacts import ArtifactStore
from alphax_agent.workflow.graph import ResearchWorkflow


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FixedBackend(ChatBackend):
    def __init__(self, response: ChatResponse | Exception) -> None:
        self.response = response

    @property
    def model(self) -> str:
        return "fixed"

    def complete(self, request: ChatRequest) -> ChatResponse:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def raw_completion(reasoning: str | None = None, reasoning_tokens: int | None = None) -> SimpleNamespace:
    """The part of a litellm response the backend reads."""
    message = SimpleNamespace(content='{"value": 1}', tool_calls=None)
    if reasoning is not None:
        message.reasoning_content = reasoning
    details = None if reasoning_tokens is None else SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=50, completion_tokens_details=details)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage, model="m")


class TestReasoning:
    backend = LiteLLMBackend(LLMSettings(model="m"))

    def test_is_taken_from_the_provider_response(self) -> None:
        response = self.backend._to_response(raw_completion("step one, step two", reasoning_tokens=40))
        assert response.content == '{"value": 1}'
        assert response.reasoning == "step one, step two"
        assert response.usage.reasoning_tokens == 40 and response.usage.completion_tokens == 50

    @pytest.mark.parametrize("raw", [raw_completion(), raw_completion("")])
    def test_is_none_when_the_provider_sends_none(self, raw: SimpleNamespace) -> None:
        response = self.backend._to_response(raw)
        assert response.reasoning is None and response.usage.reasoning_tokens == 0

    def test_survives_the_response_cache(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put("ab12", ChatResponse(content="x", reasoning="because"))
        cached = cache.get("ab12")
        assert cached is not None and cached.reasoning == "because" and cached.cached

    def test_old_cache_entries_without_it_still_load(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        (tmp_path / "cd").mkdir()
        (tmp_path / "cd" / "cd34.json").write_text('{"content": "x", "usage": {"calls": 1}}')
        cached = cache.get("cd34")
        assert cached is not None and cached.reasoning is None

    def test_usage_adds_up_reasoning_tokens(self) -> None:
        total = TokenUsage(reasoning_tokens=3) + TokenUsage(reasoning_tokens=4, completion_tokens=9)
        assert total.reasoning_tokens == 7 and total.completion_tokens == 9


class TestCallContext:
    def test_nests_and_restores(self) -> None:
        assert current_context() == {}
        with call_context(loop=1, stage="propose"):
            with call_context(stage="design", turn=0):
                assert current_context() == {"loop": 1, "stage": "design", "turn": 0}
            assert current_context() == {"loop": 1, "stage": "propose"}
        assert current_context() == {}

    def test_is_restored_after_an_error(self) -> None:
        with pytest.raises(RuntimeError), call_context(loop=1):
            raise RuntimeError
        assert current_context() == {}


class TestAuditLog:
    def client(self, response: ChatResponse | Exception, path: Path) -> LLMClient:
        client = LLMClient(FixedBackend(response), LLMSettings(model="fixed", max_retries=0), sleeper=lambda _: None)
        client.attach_audit(LLMAuditLog(path))
        return client

    def test_records_reasoning_and_context(self, tmp_path: Path) -> None:
        client = self.client(ChatResponse(content="answer", reasoning="thinking"), tmp_path / "calls.jsonl")
        with call_context(loop=2, stage="propose"):
            client.chat([ChatMessage.user("q")], agent="a")
        client.chat([ChatMessage.user("q")], agent="b")
        first, second = read_lines(tmp_path / "calls.jsonl")
        assert first["reasoning"] == "thinking" and "reasoning" not in first["response"]
        assert first["response"]["content"] == "answer"
        assert first["context"] == {"loop": 2, "stage": "propose"}
        assert second["context"] == {}

    def test_records_the_context_of_a_failed_call(self, tmp_path: Path) -> None:
        client = self.client(RuntimeError("boom"), tmp_path / "calls.jsonl")
        with pytest.raises(Exception, match="boom"), call_context(loop=3, stage="design"):
            client.chat([ChatMessage.user("q")], agent="a")
        (line,) = read_lines(tmp_path / "calls.jsonl")
        assert line["response"] is None and line["reasoning"] is None and "boom" in line["error"]
        assert line["context"] == {"loop": 3, "stage": "design"}


class Answer(BaseModel):
    value: int


class CountingPrompt(AgentPrompt):
    system_template = "t.system"
    user_template = "t.user"


class CountingAgent(Agent[Answer]):
    name = "counter"
    output_model = Answer

    def ask(self) -> Answer:
        return self._ask()


class TestConversations:
    def test_repair_rounds_share_a_conversation_and_count_turns(
        self, backend: ScriptedBackend, llm: LLMClient, tmp_path: Path
    ) -> None:
        llm.attach_audit(LLMAuditLog(tmp_path / "calls.jsonl"))
        backend.on("counter", ["not json", '{"value": 1}', '{"value": 2}'])
        agent = CountingAgent(llm, CountingPrompt(library=PromptLibrary({"t.system": "s", "t.user": "u"})))
        assert agent.ask().value == 1
        assert agent.ask().value == 2
        contexts = [line["context"] for line in read_lines(tmp_path / "calls.jsonl")]
        assert [context["turn"] for context in contexts] == [0, 1, 0]
        assert contexts[0]["conversation"] == contexts[1]["conversation"] != contexts[2]["conversation"]
        assert current_context() == {}


class AskingProposer(HypothesisProposer):
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def propose(self, trace: Trace) -> Hypothesis:
        self._llm.chat([ChatMessage.user("idea?")], agent="asker")
        return Hypothesis(statement="s", rationale="r")


class OneTaskDesigner(ExperimentDesigner):
    def design(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        return Experiment(hypothesis=hypothesis, tasks=[Task(name=f"t{len(trace)}", description="d")])


class FailsInLoopZero(Developer):
    def develop(self, experiment: Experiment, trace: Trace) -> Experiment:
        if len(trace) == 0:
            raise CoderError("nothing compiled")
        return experiment


class PassThroughRunner(Runner):
    def run(self, experiment: Experiment, trace: Trace) -> Experiment:
        return experiment


class RejectingSummarizer(Summarizer):
    def summarize(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:
        return ExperimentFeedback(decision=False, reason="no result")


class TestStepEvents:
    @pytest.fixture
    def run(self, backend: ScriptedBackend, llm: LLMClient, tmp_path: Path) -> ArtifactStore:
        backend.on("asker", lambda request: "ok")
        store = ArtifactStore(tmp_path / "run")
        llm.attach_audit(LLMAuditLog(store.llm_audit_path))
        parts = ResearchComponents(
            proposer=AskingProposer(llm),
            designer=OneTaskDesigner(),
            developer=FailsInLoopZero(),
            runner=PassThroughRunner(),
            summarizer=RejectingSummarizer(),
        )
        ResearchWorkflow(parts, WorkflowSettings(), store).run(loops=2)
        return store

    def test_every_step_is_recorded_with_its_outcome(self, run: ArtifactStore) -> None:
        events = read_lines(run.events_path)
        sequence = [(event["loop"], event["stage"], event["event"]) for event in events]
        assert [item for item in sequence if item[0] == 0] == [
            (0, "baseline", "started"),
            (0, "baseline", "finished"),
            (0, "propose", "started"),
            (0, "propose", "finished"),
            (0, "design", "started"),
            (0, "design", "finished"),
            (0, "develop", "started"),
            (0, "develop", "failed"),
            (0, "summarize", "started"),
            (0, "summarize", "finished"),
            (0, "record", "started"),
            (0, "record", "finished"),
        ]
        assert [item[1:] for item in sequence if item[0] == 1] == [
            (stage, event)
            for stage in ("propose", "design", "develop", "run", "summarize", "record")
            for event in ("started", "finished")
        ]

    def test_a_failed_step_carries_its_error_and_every_end_a_duration(self, run: ArtifactStore) -> None:
        events = read_lines(run.events_path)
        (failed,) = [event for event in events if event["event"] == "failed"]
        assert (failed["loop"], failed["stage"], failed["error_type"], failed["message"]) == (
            0,
            "develop",
            "CoderError",
            "nothing compiled",
        )
        assert all(event["seconds"] >= 0 for event in events if event["event"] != "started")
        assert all("time" in event for event in events)

    def test_model_calls_carry_the_loop_and_stage(self, run: ArtifactStore) -> None:
        contexts = [line["context"] for line in read_lines(run.llm_audit_path)]
        assert contexts == [{"loop": 0, "stage": "propose"}, {"loop": 1, "stage": "propose"}]


DATA_ROOT = "no-data"
HYPOTHESIS = {"statement": "s", "rationale": "r"}
STRATEGIES = {"strategies": [{"name": "A", "description": "[Mean Reversion] a rule"}]}


class TestRunRecord:
    @pytest.fixture
    def app(
        self, backend: ScriptedBackend, llm: LLMClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> AlphaXApplication:
        backend.on("strategy_proposer", lambda request: json.dumps(HYPOTHESIS))
        backend.on("strategy_designer", lambda request: json.dumps(STRATEGIES))
        cache = SyntheticJFinance().write_catalog(tmp_path / "cache")
        options = {"alphax": {"data_dir": str(tmp_path / DATA_ROOT)}, "cache_dir": str(cache)}
        scenario = CryptoStrategyScenario(CryptoStrategySettings.model_validate(options))
        # Only the run record is under test: no alphax interpreter, the panels are already there.
        monkeypatch.setattr(scenario, "prepare", lambda *, force=False: None)
        settings = AlphaXSettings(workflow=WorkflowSettings(run_root=tmp_path / "runs", run_name="r1"))
        return AlphaXApplication(settings, llm=llm, scenario=scenario)

    def test_a_finished_run_says_what_it_was(self, app: AlphaXApplication, tmp_path: Path) -> None:
        report = app.run(until="design")
        record = json.loads((report.run_dir / "run.json").read_text())
        assert record["status"] == "finished" and record["finished_at"] is not None
        assert record["scenario"] == "crypto_strategy" and record["pid"] == os.getpid()
        assert record["options"] == {"loops": None, "until": "design", "resume": False} and record["resumes"] == 0
        calls = read_lines(report.run_dir / "llm_calls.jsonl")
        assert [(line["agent"], line["context"]["loop"], line["context"]["stage"]) for line in calls] == [
            ("strategy_proposer", 0, "propose"),
            ("strategy_designer", 0, "design"),
        ]
        assert all(line["context"]["turn"] == 0 and line["context"]["conversation"] for line in calls)

    @pytest.mark.parametrize(("error", "status"), [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")])
    def test_an_aborted_run_says_how(
        self, app: AlphaXApplication, monkeypatch: pytest.MonkeyPatch, error: type[BaseException], status: str
    ) -> None:
        def prepare(*, force: bool = False) -> None:
            raise error

        monkeypatch.setattr(app.scenario, "prepare", prepare)
        with pytest.raises(error):
            app.run(until="design")
        record = ArtifactStore.open(app.settings.workflow.run_root / "r1").load_run()
        assert record is not None and record["status"] == status and record["finished_at"] is not None

    def test_a_resumed_run_keeps_its_start_time(self, app: AlphaXApplication) -> None:
        first = app.run(until="design")
        started = json.loads((first.run_dir / "run.json").read_text())["started_at"]
        second = app.run(resume=first.run_dir, until="design")
        record = json.loads((second.run_dir / "run.json").read_text())
        assert record["started_at"] == started and record["resumes"] == 1
        assert record["options"]["resume"] is True and record["status"] == "finished"
