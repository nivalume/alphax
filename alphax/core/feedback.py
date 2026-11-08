"""Feedback at the two levels of the research loop.

* :class:`ImplementationFeedback` judges one implementation of one task (inner CoSTEER loop).
* :class:`ExperimentFeedback` judges a whole experiment against the current best (outer loop).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field


class StageResult(BaseModel):
    """Outcome of one evaluation stage."""

    stage: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Quality of the stage outcome in [0, 1]")
    message: str = ""
    skipped: bool = False

    def render(self) -> str:
        status = "SKIPPED" if self.skipped else ("PASS" if self.passed else "FAIL")
        return f"[{self.stage}] {status} (score {self.score:.2f})\n{self.message}".rstrip()


class ImplementationFeedback(BaseModel):
    """Aggregated verdict of an evaluation pipeline on one implementation."""

    stages: list[StageResult] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    accepted: bool = False

    @classmethod
    def from_stages(cls, stages: Sequence[StageResult], weights: Sequence[float]) -> ImplementationFeedback:
        total = sum(weights) or 1.0
        score = sum(weight * stage.score for stage, weight in zip(stages, weights, strict=True) if not stage.skipped)
        accepted = bool(stages) and all(stage.passed and not stage.skipped for stage in stages)
        return cls(stages=list(stages), score=min(1.0, score / total), accepted=accepted)

    def stage(self, name: str) -> StageResult | None:
        return next((stage for stage in self.stages if stage.stage == name), None)

    def first_failure(self) -> StageResult | None:
        return next((stage for stage in self.stages if not stage.passed and not stage.skipped), None)

    def render(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        body = "\n\n".join(stage.render() for stage in self.stages)
        return f"Verdict: {verdict} (score {self.score:.2f})\n\n{body}"


class StepFailure(BaseModel):
    """A research step that raised instead of producing its output."""

    stage: str
    error_type: str
    message: str


class ExperimentFeedback(BaseModel):
    """Verdict on an experiment.

    ``hypothesis_supported`` is the research verdict (did the evidence support the idea),
    ``decision`` is the promotion verdict (does the experiment become the new best).
    The two differ, e.g. a supported hypothesis whose factors duplicate the current best.
    """

    decision: bool
    reason: str
    observations: str = ""
    hypothesis_evaluation: str = ""
    hypothesis_supported: bool | None = None
    next_hypothesis: str = ""
    failure: StepFailure | None = None

    @classmethod
    def from_failure(cls, failure: StepFailure) -> ExperimentFeedback:
        return cls(
            decision=False,
            reason=f"The {failure.stage} step failed with {failure.error_type}: {failure.message}",
            hypothesis_supported=None,
            failure=failure,
        )

    @property
    def failed(self) -> bool:
        return self.failure is not None

    def render(self) -> str:
        lines = [f"Decision: {'promoted to best' if self.decision else 'not promoted'}", f"Reason: {self.reason}"]
        if self.hypothesis_supported is not None:
            lines.append(f"Hypothesis supported: {self.hypothesis_supported}")
        for label, value in (
            ("Observations", self.observations),
            ("Hypothesis evaluation", self.hypothesis_evaluation),
            ("Suggested next hypothesis", self.next_hypothesis),
        ):
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)
