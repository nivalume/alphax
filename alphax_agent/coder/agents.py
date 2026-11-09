"""The two agents of the coder: one writes code, one distills lessons from attempts."""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, SerializeAsAny

from alphax_agent.agents.agent import Agent
from alphax_agent.agents.tool import Tool
from alphax_agent.coder.lessons import ErrorSignature, Lesson, LessonBook, LessonDraft
from alphax_agent.coder.tree import Operator, SolutionNode, SolutionTree
from alphax_agent.core.exceptions import OutputParsingError
from alphax_agent.core.prompts import PromptLibrary
from alphax_agent.core.scenario import ScenarioContext
from alphax_agent.core.task import Task
from alphax_agent.core.text import Excerpt
from alphax_agent.llm.client import LLMClient


class CodeSolution(BaseModel):
    rationale: str = Field(description="How the code implements the task and what changed relative to the parent")
    files: dict[str, str] = Field(description="Map of file name to the complete content of that file")


class ImplementationRequest(BaseModel):
    task: SerializeAsAny[Task]
    operator: Operator
    parent: SolutionNode | None = None
    lineage: list[SolutionNode] = Field(default_factory=list, description="Ancestors of the parent, oldest first")
    lessons: list[Lesson] = Field(default_factory=list)
    peers: list[SolutionNode] = Field(default_factory=list)
    peer_names: dict[str, str] = Field(default_factory=dict, description="Task id to task name for the peers")


class CoderAgent(Agent[CodeSolution]):
    """Writes the files of one task, as a draft, a repair of a failed parent or an improvement.

    Scenarios subclass it to set ``required_files`` and their tools, and override the
    ``coder.*`` templates in their own prompt file when the generic ones are not enough.
    """

    name = "coder"
    output_model = CodeSolution
    system_template = "coder.system"
    user_template = "coder.user"
    required_files: ClassVar[tuple[str, ...]] = ()
    feedback_limit: ClassVar[int] = 6000

    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptLibrary,
        context: ScenarioContext,
        tools: Sequence[Tool] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, PromptLibrary.from_package("alphax_agent.coder").merged(prompts), tools, **kwargs)
        self._context = context

    def implement(self, request: ImplementationRequest) -> CodeSolution:
        parent = request.parent
        return self._ask(
            scenario=self._context.render(),
            task=request.task.render(),
            operator=request.operator.value,
            required_files=list(self.required_files),
            parent_code=parent.workspace.render_code() if parent else "",
            parent_feedback=Excerpt.middle(parent.feedback.render(), self.feedback_limit) if parent else "",
            lineage=[self._summarize(node) for node in request.lineage],
            lessons=LessonBook.render(request.lessons),
            peers=[
                {"task": request.peer_names.get(node.task_id, node.task_id), "code": node.workspace.render_code()}
                for node in request.peers
            ],
            tools=self.tools,
        )

    def _validate(self, output: CodeSolution) -> CodeSolution:
        missing = [name for name in self.required_files if not output.files.get(name, "").strip()]
        if missing:
            raise OutputParsingError(f"The reply must contain the complete content of {missing}.")
        unsafe = [name for name in output.files if name.startswith("/") or ".." in name.split("/")]
        if unsafe:
            raise OutputParsingError(f"File names must be relative paths inside the workspace: {unsafe}")
        return output

    @staticmethod
    def _summarize(node: SolutionNode) -> dict[str, Any]:
        failure = node.feedback.first_failure()
        return {
            "operator": node.operator.value,
            "score": round(node.score, 3),
            "verdict": "accepted" if node.accepted else "rejected",
            "issue": Excerpt.middle(failure.message, 600) if failure else "",
            "rationale": Excerpt.middle(node.rationale, 400),
        }


class TransitionKind(StrEnum):
    FIXED = "fixed"
    REGRESSED = "regressed"
    STILL_FAILING = "still_failing"
    FAILED_DRAFT = "failed_draft"
    IMPROVED = "improved"


class Transition(BaseModel):
    """What one attempt changed, in the form the reflection agent reasons about."""

    task_name: str
    kind: TransitionKind
    signature: str
    detail: str

    PRIORITY: ClassVar[dict[TransitionKind, int]] = {
        TransitionKind.FIXED: 0,
        TransitionKind.REGRESSED: 1,
        TransitionKind.STILL_FAILING: 2,
        TransitionKind.FAILED_DRAFT: 3,
        TransitionKind.IMPROVED: 4,
    }

    @classmethod
    def observe(cls, tree: SolutionTree, node: SolutionNode) -> Transition | None:
        """The transition from the parent to ``node``, or ``None`` when it teaches nothing."""
        parent = tree.get(node.parent_id) if node.parent_id is not None else None
        kind = cls._classify(parent, node)
        if kind is None:
            return None
        failing = node if not node.accepted else parent
        signature = ErrorSignature.from_feedback(failing.feedback if failing else None)
        if parent is None:
            detail = (
                f"Code:\n{node.workspace.render_code()}\n\nFeedback:\n{Excerpt.middle(node.feedback.render(), 3000)}"
            )
        else:
            detail = (
                f"Parent feedback:\n{Excerpt.middle(parent.feedback.render(), 2500)}\n\n"
                f"Change (unified diff):\n{Excerpt.middle(cls._diff(parent, node), 3000)}\n\n"
                f"Result feedback:\n{Excerpt.middle(node.feedback.render(), 2500)}"
            )
        return cls(
            task_name=tree.task.name, kind=kind, signature=signature.key if signature else "general", detail=detail
        )

    @staticmethod
    def _classify(parent: SolutionNode | None, node: SolutionNode) -> TransitionKind | None:
        if parent is None:
            return None if node.accepted else TransitionKind.FAILED_DRAFT
        if not parent.accepted and node.accepted:
            return TransitionKind.FIXED
        if parent.accepted and not node.accepted:
            return TransitionKind.REGRESSED
        if not parent.accepted:
            same = ErrorSignature.from_feedback(parent.feedback) == ErrorSignature.from_feedback(node.feedback)
            return TransitionKind.STILL_FAILING if same else None
        return TransitionKind.IMPROVED if node.score > parent.score else None

    @staticmethod
    def _diff(parent: SolutionNode, node: SolutionNode) -> str:
        chunks: list[str] = []
        for name in sorted(set(parent.files) | set(node.files)):
            before = parent.files.get(name, "").splitlines(keepends=True)
            after = node.files.get(name, "").splitlines(keepends=True)
            chunks.extend(difflib.unified_diff(before, after, fromfile=f"parent/{name}", tofile=f"child/{name}", n=2))
        return "".join(chunks) or "(no change)"


class ReflectionOutput(BaseModel):
    lessons: list[LessonDraft] = Field(default_factory=list, description="New lessons, at most three")
    reinforced: list[str] = Field(default_factory=list, description="Ids of existing lessons the transitions confirm")


class ReflectionAgent(Agent[ReflectionOutput]):
    """Turns transitions into reusable lessons, shared by every task of the run."""

    name = "reflector"
    output_model = ReflectionOutput
    system_template = "reflector.system"
    user_template = "reflector.user"
    max_lessons: ClassVar[int] = 3

    def __init__(self, llm: LLMClient, prompts: PromptLibrary, context: ScenarioContext, **kwargs: Any) -> None:
        super().__init__(llm, PromptLibrary.from_package("alphax_agent.coder").merged(prompts), **kwargs)
        self._context = context

    def reflect(self, transitions: Sequence[Transition], existing: Sequence[Lesson]) -> ReflectionOutput:
        output = self._ask(
            scenario=self._context.render(["background", "interface", "output_format"]),
            transitions=[transition.model_dump() for transition in transitions],
            lessons=LessonBook.render(existing),
            max_lessons=self.max_lessons,
        )
        known = {lesson.id for lesson in existing}
        return ReflectionOutput(
            lessons=[draft for draft in output.lessons if draft.text.strip()][: self.max_lessons],
            reinforced=[lesson_id for lesson_id in output.reinforced if lesson_id in known],
        )
