"""CoSTEER: the developer role, implemented as collaborative feedback-guided tree search."""

from __future__ import annotations

import logging
from pathlib import Path

from alphax_agent.coder.agents import CoderAgent, ReflectionAgent
from alphax_agent.coder.evaluation import EvaluationPipeline
from alphax_agent.coder.graph import CoSTEERGraph, SearchSession
from alphax_agent.coder.lessons import LessonBook
from alphax_agent.coder.policy import SearchPolicy
from alphax_agent.config.settings import CoSTEERSettings
from alphax_agent.core.exceptions import CoderError
from alphax_agent.core.experiment import Experiment, Implementation
from alphax_agent.core.roles import Developer
from alphax_agent.core.trace import Trace
from alphax_agent.core.workspace import WorkspaceFactory

logger = logging.getLogger(__name__)


class CoSTEER(Developer):
    """Implements every task of an experiment by searching over candidate implementations.

    Compared with retrieval-based CoSTEER, knowledge is carried by three mechanisms that need
    no embedding index:

    * the solution tree of each task (repairs see the failed parent and its lineage),
    * accepted implementations of sibling tasks, shown to the coder as working conventions,
    * the lesson book, written by the reflection agent and matched by error signature.

    The lesson book outlives a single experiment, so later loops start from what earlier
    loops learned; with ``lesson_store`` set it also outlives the run.
    """

    def __init__(
        self,
        coder: CoderAgent,
        pipeline: EvaluationPipeline,
        workspaces: WorkspaceFactory,
        settings: CoSTEERSettings,
        workspace_root: Path,
        *,
        reflector: ReflectionAgent | None = None,
        lessons: LessonBook | None = None,
    ) -> None:
        self._settings = settings
        self._root = Path(workspace_root)
        if lessons is not None:
            self._lessons = lessons
        elif settings.lesson_store is not None:
            self._lessons = LessonBook.load(settings.lesson_store)
        else:
            self._lessons = LessonBook()
        self._graph = CoSTEERGraph(coder, pipeline, workspaces, SearchPolicy.create(settings), settings, reflector)

    @property
    def lessons(self) -> LessonBook:
        return self._lessons

    def develop(self, experiment: Experiment, trace: Trace) -> Experiment:
        if not experiment.tasks:
            raise CoderError("The experiment has no task to implement.")
        session = SearchSession(experiment, self._lessons, self._root / experiment.id)
        try:
            self._graph.run(session)
        finally:
            session.save_summary()
            self._persist_lessons()
        for task_id, node in session.forest.best_nodes().items():
            experiment.implementations[task_id] = Implementation(
                task_id=task_id, workspace=node.workspace, feedback=node.feedback
            )
        if not experiment.accepted():
            failures = [
                f"- {task.name}: {failure.stage}: {failure.message[:300]}"
                for task in experiment.tasks
                if (implementation := experiment.implementations.get(task.id)) is not None
                and (failure := implementation.feedback.first_failure()) is not None
            ]
            raise CoderError("No task passed evaluation.\n" + "\n".join(failures))
        logger.info(
            "CoSTEER accepted %d/%d task(s): %s",
            len(experiment.accepted()),
            len(experiment.tasks),
            session.forest.summary(),
        )
        return experiment

    def _persist_lessons(self) -> None:
        targets = [self._root / "lessons.json"]
        if self._settings.lesson_store is not None:
            targets.append(Path(self._settings.lesson_store))
        for target in targets:
            self._lessons.save(target)
