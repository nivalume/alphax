from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import ScriptedBackend

from alphax_agent.coder import (
    CoderAgent,
    CoderPrompt,
    CoSTEER,
    ErrorSignature,
    EvaluationPipeline,
    EvaluationStage,
    Expansion,
    LessonBook,
    LessonDraft,
    Operator,
    ReflectionAgent,
    ReflectorPrompt,
    SearchPolicy,
    SolutionNode,
    SolutionTree,
)
from alphax_agent.coder.evaluation import EvaluationContext
from alphax_agent.config.settings import CoSTEERSettings
from alphax_agent.core.exceptions import CoderError
from alphax_agent.core.experiment import Experiment
from alphax_agent.core.feedback import ImplementationFeedback, StageResult
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.task import Task
from alphax_agent.core.workspace import Workspace, WorkspaceFactory
from alphax_agent.llm.client import LLMAuditLog
from alphax_agent.llm.context import call_context, current_context

CODER_PROMPT = CoderPrompt(library=PromptLibrary())
REFLECTOR_PROMPT = ReflectorPrompt(library=PromptLibrary())


def feedback(accepted: bool, score: float, message: str = "") -> ImplementationFeedback:
    stage = StageResult(stage="run", passed=accepted, score=score, message=message)
    return ImplementationFeedback(stages=[stage], score=score, accepted=accepted)


def node(
    task: Task,
    accepted: bool,
    score: float,
    parent: SolutionNode | None = None,
    message: str = "",
    round_index: int = 0,
) -> SolutionNode:
    return SolutionNode(
        task_id=task.id,
        parent_id=parent.id if parent else None,
        operator=Operator.DRAFT if parent is None else (Operator.IMPROVE if parent.accepted else Operator.REPAIR),
        depth=parent.depth + 1 if parent else 0,
        round=round_index,
        workspace=Workspace(path=Path("/tmp/none")),
        feedback=feedback(accepted, score, message),
    )


class TestSolutionTree:
    def test_lineage_streak_and_best(self) -> None:
        task = Task(name="t", description="d")
        tree = SolutionTree(task)
        root = node(task, False, 0.1)
        child = node(task, False, 0.2, root, round_index=1)
        grandchild = node(task, True, 0.8, child, round_index=2)
        for item in (root, child, grandchild):
            tree.add(item)
        assert [n.id for n in tree.ancestors(grandchild)] == [root.id, child.id]
        assert tree.failure_streak(child) == 2
        assert tree.failure_streak(grandchild) == 0
        assert tree.best() is grandchild
        assert tree.get(root.id).expansions == 1
        assert tree.solved(0.8) and not tree.solved(0.9)


class TestSearchPolicy:
    settings = CoSTEERSettings(
        policy="greedy", num_drafts=1, max_repair_depth=2, target_score=0.9, max_attempts_per_task=5
    )

    def test_draft_repair_improve_and_stop(self) -> None:
        task = Task(name="t", description="d")
        tree = SolutionTree(task)
        policy = SearchPolicy.create(self.settings)
        assert policy.next_expansion(tree) == Expansion(task_id=task.id, operator=Operator.DRAFT)
        failed = node(task, False, 0.3)
        tree.add(failed)
        assert policy.next_expansion(tree) == Expansion(task_id=task.id, operator=Operator.REPAIR, parent_id=failed.id)
        weak = node(task, True, 0.7, failed, round_index=1)
        tree.add(weak)
        assert policy.next_expansion(tree) == Expansion(task_id=task.id, operator=Operator.IMPROVE, parent_id=weak.id)
        tree.add(node(task, True, 0.95, weak, round_index=2))
        assert policy.next_expansion(tree) is None

    def test_dead_branch_triggers_new_draft(self) -> None:
        task = Task(name="t", description="d")
        tree = SolutionTree(task)
        first = node(task, False, 0.1)
        second = node(task, False, 0.1, first, round_index=1)
        tree.add(first)
        tree.add(second)
        expansion = SearchPolicy.create(self.settings).next_expansion(tree)
        assert expansion is not None and expansion.operator is Operator.DRAFT

    def test_budget(self) -> None:
        task = Task(name="t", description="d")
        tree = SolutionTree(task)
        parent = None
        for index in range(5):
            parent = node(task, False, 0.1, None if index % 2 == 0 else parent, round_index=index)
            tree.add(parent)
        assert SearchPolicy.create(self.settings).next_expansion(tree) is None

    def test_ucb_prefers_less_explored_node(self) -> None:
        settings = CoSTEERSettings(
            policy="ucb", exploration=1.0, num_drafts=2, target_score=0.99, max_attempts_per_task=10
        )
        task = Task(name="t", description="d")
        tree = SolutionTree(task)
        explored, fresh = node(task, True, 0.8), node(task, True, 0.75)
        tree.add(explored)
        tree.add(fresh)
        for index in range(3):
            tree.add(node(task, False, 0.1, explored, round_index=index + 1))
        expansion = SearchPolicy.create(settings).next_expansion(tree)
        assert expansion is not None and expansion.parent_id == fresh.id
        greedy = SearchPolicy.create(settings.model_copy(update={"policy": "greedy"})).next_expansion(tree)
        assert greedy is not None and greedy.parent_id == explored.id

    def test_repair_follows_the_leaf(self) -> None:
        task = Task(name="t", description="d")
        tree = SolutionTree(task)
        first = node(task, False, 0.1)
        tree.add(first)
        tree.add(second := node(task, False, 0.2, first, round_index=1))
        settings = self.settings.model_copy(update={"max_repair_depth": 3})
        expansion = SearchPolicy.create(settings).next_expansion(tree)
        assert expansion is not None and expansion.parent_id == second.id


class TestLessons:
    def test_signature_from_feedback(self) -> None:
        fb = feedback(False, 0.0, "Traceback ...\nKeyError: 'close'")
        assert ErrorSignature.from_feedback(fb) == ErrorSignature("run", "KeyError")
        assert ErrorSignature.from_feedback(feedback(True, 1.0)) is None
        assert ErrorSignature.parse("run:KeyError") == ErrorSignature("run", "KeyError")

    def test_dedupe_ranking_and_persistence(self, tmp_path: Path) -> None:
        book = LessonBook()
        key_error = book.add(
            LessonDraft(text="Columns keep the $ prefix: use $close.", signature="run:KeyError"), "a", 0
        )
        book.add(LessonDraft(text="columns keep the $ prefix, use $close", signature="run:KeyError"), "b", 1)
        general = book.add(LessonDraft(text="Group by instrument before rolling."), "a", 0)
        book.reinforce([general.id, general.id, "unknown"])
        assert len(book) == 2
        by_error = book.select(ErrorSignature("run", "KeyError"), 2)
        assert by_error[0].id == key_error.id and by_error[0].support == 2
        assert book.select(None, 1)[0].id == general.id
        book.save(tmp_path / "lessons.json")
        assert {lesson.id for lesson in LessonBook.load(tmp_path / "lessons.json").all()} == {key_error.id, general.id}


class ContainsStage(EvaluationStage):
    name = "run"

    def evaluate(self, context: EvaluationContext) -> StageResult:
        code = context.workspace.files.get("main.py", "")
        if "fixed" in code:
            return self.result(True, 1.0, "ok")
        return self.result(False, 0.1, "Traceback (most recent call last):\nKeyError: 'close'")


class PlainFactory(WorkspaceFactory):
    def create(self, task: Task, path: Path) -> Workspace:
        return Workspace(path=path)


class ToyCoder(CoderAgent):
    required_files = ("main.py",)


def coder_reply(request) -> str:
    user = request.messages[1].content
    code = "fixed" if "Mode: repair" in user else "broken"
    return json.dumps({"rationale": code, "files": {"main.py": f"print('{code}')"}})


class TestCoSTEER:
    def test_search_repairs_and_shares_lessons(self, backend: ScriptedBackend, llm, tmp_path: Path) -> None:
        backend.on("coder", coder_reply)
        backend.on(
            "reflector",
            [
                json.dumps(
                    {
                        "lessons": [{"text": "Columns are prefixed with $; read $close.", "signature": "run:KeyError"}],
                        "reinforced": [],
                    }
                )
            ]
            + [json.dumps({"lessons": [], "reinforced": []})] * 5,
        )
        settings = CoSTEERSettings(policy="greedy", max_rounds=3, target_score=0.9, lesson_store=tmp_path / "kb.json")
        developer = CoSTEER(
            ToyCoder(llm, CODER_PROMPT),
            EvaluationPipeline([ContainsStage()]),
            PlainFactory(),
            settings,
            tmp_path / "coding",
            reflector=ReflectionAgent(llm, REFLECTOR_PROMPT),
        )
        experiment = Experiment(tasks=[Task(name="alpha", description="a"), Task(name="beta", description="b")])
        developer.develop(experiment, None)

        assert [task.name for task, _ in experiment.accepted()] == ["alpha", "beta"]
        coder_requests = backend.requests_of("coder")
        assert len(coder_requests) == 4
        repairs = [
            request.messages[1].content for request in coder_requests if "Mode: repair" in request.messages[1].content
        ]
        assert len(repairs) == 2 and all("Columns are prefixed with $" in text for text in repairs)
        assert "KeyError" in repairs[0]
        assert (tmp_path / "kb.json").exists() and len(LessonBook.load(tmp_path / "kb.json")) == 1
        tree = json.loads((tmp_path / "coding" / experiment.id / "search_tree.json").read_text())
        assert [entry["operator"] for entry in tree["alpha"]] == ["draft", "repair"]
        assert (
            Path(experiment.implementations[experiment.tasks[0].id].workspace.path, "main.py").read_text()
            == "print('fixed')"
        )

    def test_model_calls_are_tagged_with_the_task_and_the_attempt(
        self, backend: ScriptedBackend, llm, tmp_path: Path
    ) -> None:
        backend.on("coder", coder_reply)
        backend.on("reflector", lambda request: json.dumps({"lessons": [], "reinforced": []}))
        llm.attach_audit(LLMAuditLog(tmp_path / "calls.jsonl"))
        developer = CoSTEER(
            ToyCoder(llm, CODER_PROMPT),
            EvaluationPipeline([ContainsStage()]),
            PlainFactory(),
            CoSTEERSettings(policy="greedy", max_rounds=3, target_score=0.9),
            tmp_path / "coding",
            reflector=ReflectionAgent(llm, REFLECTOR_PROMPT),
        )
        experiment = Experiment(tasks=[Task(name="alpha", description="a"), Task(name="beta", description="b")])
        with call_context(loop=4, stage="develop"):
            developer.develop(experiment, None)
        assert current_context() == {}

        lines = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        coder = [line["context"] for line in lines if line["agent"] == "coder"]
        # The attempts run in worker threads; their calls still know the loop, stage, task and node.
        assert {(c["loop"], c["stage"], c["task"], c["operator"], c["round"]) for c in coder} == {
            (4, "develop", "alpha", "draft", 0),
            (4, "develop", "beta", "draft", 0),
            (4, "develop", "alpha", "repair", 1),
            (4, "develop", "beta", "repair", 1),
        }
        assert {c["task_id"] for c in coder} == {task.id for task in experiment.tasks}
        tree = json.loads((tmp_path / "coding" / experiment.id / "search_tree.json").read_text())
        assert {c["node_id"] for c in coder} == {entry["id"] for entries in tree.values() for entry in entries}
        assert {c["parent_id"] for c in coder if c["operator"] == "repair"} <= {c["node_id"] for c in coder}
        reflections = [line["context"] for line in lines if line["agent"] == "reflector"]
        assert reflections and all("round" in c and c["loop"] == 4 for c in reflections)

    def test_raises_when_nothing_is_accepted(self, backend: ScriptedBackend, llm, tmp_path: Path) -> None:
        backend.on("coder", lambda request: json.dumps({"rationale": "", "files": {"main.py": "broken"}}))
        developer = CoSTEER(
            ToyCoder(llm, CODER_PROMPT),
            EvaluationPipeline([ContainsStage()]),
            PlainFactory(),
            CoSTEERSettings(max_rounds=2),
            tmp_path,
        )
        experiment = Experiment(tasks=[Task(name="alpha", description="a")])
        with pytest.raises(CoderError, match="KeyError"):
            developer.develop(experiment, None)
        assert experiment.implementations and not experiment.accepted()


class ToyScenarioCoderPrompt(CoderPrompt):
    """A scenario's coder prompt: the generic inputs, plus a section of its own."""

    interface: str

    def sections(self) -> dict[str, str]:
        return {"interface": self.interface}


class TestCoderPrompt:
    INPUTS = dict(
        task="Task: alpha",
        operator="draft",
        required_files=["main.py"],
        parent_code="",
        parent_feedback="",
        lineage=[],
        lessons="(none)",
        peers=[],
        tools=[],
    )

    def test_the_generic_prompt_describes_no_domain(self) -> None:
        system, user = CODER_PROMPT.format_messages(**self.INPUTS)
        assert "senior quantitative developer" in system.text and "main.py" in system.text
        assert "described above" not in system.text
        assert "# Mode: draft" in user.text

    def test_a_scenario_overrides_the_system_template_and_adds_its_sections(self) -> None:
        library = PromptLibrary(
            {"coder.system": "Write {{ required_files | join(', ') }}.\n### Interface\n{{ interface }}"}
        )
        system, user = ToyScenarioCoderPrompt(library=library, interface="print the answer").format_messages(
            **self.INPUTS
        )
        assert system.text == "Write main.py.\n### Interface\nprint the answer"
        # The user template is still the package's own.
        assert "# Mode: draft" in user.text and "Task: alpha" in user.text
