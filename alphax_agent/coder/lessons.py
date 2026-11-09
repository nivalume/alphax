"""Lessons: knowledge the coder accumulates without retrieval.

A lesson is a short, task-independent rule distilled by the reflection agent from an
observed transition (a failure that was fixed, a failure that persisted). Lessons are
indexed by the deterministic :class:`ErrorSignature` of the failure they address and
ranked by how often they were confirmed, so the coder sees the rules relevant to the error
in front of it, with no embedding search involved.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from alphax_agent.core.feedback import ImplementationFeedback


@dataclass(frozen=True)
class ErrorSignature:
    """Class of a failure: the failing evaluation stage and the exception type, if any."""

    stage: str
    exception: str | None = None

    _EXCEPTION = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception))\b")
    GENERAL = "general"

    @classmethod
    def from_feedback(cls, feedback: ImplementationFeedback | None) -> ErrorSignature | None:
        if feedback is None:
            return None
        failure = feedback.first_failure()
        if failure is None:
            return None
        found = cls._EXCEPTION.findall(failure.message)
        return cls(stage=failure.stage, exception=found[-1] if found else None)

    @classmethod
    def parse(cls, key: str) -> ErrorSignature | None:
        if not key or key == cls.GENERAL:
            return None
        stage, _, exception = key.partition(":")
        return cls(stage=stage, exception=exception or None)

    @property
    def key(self) -> str:
        return f"{self.stage}:{self.exception}" if self.exception else self.stage

    def affinity(self, other: ErrorSignature | None) -> int:
        """2 for the same stage and exception, 1 for the same stage, 0 otherwise."""
        if other is None or other.stage != self.stage:
            return 0
        return 2 if other.exception == self.exception else 1


class Lesson(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    text: str
    signature: str = ErrorSignature.GENERAL
    support: int = 1
    origin_task: str = ""
    created_round: int = 0

    @property
    def parsed_signature(self) -> ErrorSignature | None:
        return ErrorSignature.parse(self.signature)


class LessonDraft(BaseModel):
    text: str = Field(description="One reusable rule, independent of the specific task, in one or two sentences")
    signature: str = Field(
        default=ErrorSignature.GENERAL,
        description="Signature of the failure the rule prevents, copied from the transition, or 'general'",
    )


class LessonBook:
    """Thread-safe store of lessons with deterministic ranking."""

    def __init__(self, lessons: Iterable[Lesson] = ()) -> None:
        self._lessons: dict[str, Lesson] = {}
        self._by_text: dict[str, str] = {}
        self._lock = threading.Lock()
        for lesson in lessons:
            self._insert(lesson)

    def __len__(self) -> int:
        return len(self._lessons)

    def all(self) -> list[Lesson]:
        with self._lock:
            return [lesson.model_copy() for lesson in self._lessons.values()]

    def add(self, draft: LessonDraft, origin_task: str, round_index: int) -> Lesson:
        """Add a lesson; a lesson already present (same normalized text) gains support instead."""
        key = self._normalize(draft.text)
        with self._lock:
            existing_id = self._by_text.get(key)
            if existing_id is not None:
                lesson = self._lessons[existing_id]
                lesson.support += 1
                return lesson.model_copy()
            lesson = Lesson(
                text=draft.text.strip(),
                signature=draft.signature or ErrorSignature.GENERAL,
                origin_task=origin_task,
                created_round=round_index,
            )
            self._insert(lesson)
            return lesson.model_copy()

    def reinforce(self, lesson_ids: Iterable[str]) -> None:
        with self._lock:
            for lesson_id in lesson_ids:
                if lesson_id in self._lessons:
                    self._lessons[lesson_id].support += 1

    def select(self, signature: ErrorSignature | None, limit: int) -> list[Lesson]:
        """The ``limit`` most relevant lessons for a failure (or for a first draft when ``None``)."""
        if limit <= 0:
            return []
        with self._lock:
            lessons = list(self._lessons.values())

        def rank(lesson: Lesson) -> tuple[int, int, int]:
            own = lesson.parsed_signature
            affinity = own.affinity(signature) if own is not None else 0
            return affinity, lesson.support, lesson.created_round

        return [lesson.model_copy() for lesson in sorted(lessons, key=rank, reverse=True)[:limit]]

    @staticmethod
    def render(lessons: Iterable[Lesson]) -> str:
        lines = [
            f"- [{lesson.id}] ({lesson.signature}, confirmed {lesson.support}x) {lesson.text}" for lesson in lessons
        ]
        return "\n".join(lines) if lines else "(no lessons yet)"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [lesson.model_dump(mode="json") for lesson in self.all()]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LessonBook:
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(Lesson.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8")))

    def _insert(self, lesson: Lesson) -> None:
        self._lessons[lesson.id] = lesson
        self._by_text[self._normalize(lesson.text)] = lesson.id

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(re.sub(r"[^\w\s$]", " ", text.casefold()).split())
