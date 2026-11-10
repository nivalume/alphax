"""The LLM-backed roles of the factor scenario."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from alphax_agent.agents.agent import Agent
from alphax_agent.agents.tool import Tool
from alphax_agent.coder.agents import CoderAgent
from alphax_agent.core.exceptions import OutputParsingError, RunnerError
from alphax_agent.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax_agent.core.feedback import ExperimentFeedback
from alphax_agent.core.promotion import PromotionRule
from alphax_agent.core.roles import ExperimentDesigner, HypothesisProposer, Summarizer
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace
from alphax_agent.core.workspace import Workspace
from alphax_agent.llm.client import LLMClient
from alphax_agent.scenarios.qlib_factor.library import FactorLibrary
from alphax_agent.scenarios.qlib_factor.prompts import (
    FactorAnalystPrompt,
    FactorCriticPrompt,
    FactorDesignerPrompt,
    FactorProposerPrompt,
)
from alphax_agent.scenarios.qlib_factor.task import FactorExperiment, FactorSpec, FactorTask

#: Metrics shown to the agents, in display order.
KEY_METRICS = (
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
)


class MetricComparison:
    """Side-by-side table of a result and the result it is compared with."""

    def __init__(
        self,
        candidate: ExperimentResult | None,
        reference: ExperimentResult | None,
        *,
        labels: tuple[str, str] = ("current best", "this experiment"),
        metrics: Sequence[str] = KEY_METRICS,
    ) -> None:
        self._candidate = candidate
        self._reference = reference
        self._labels = labels
        self._metrics = metrics

    def render(self) -> str:
        reference_label, candidate_label = self._labels
        rows = [f"| metric | {reference_label} | {candidate_label} | change |", "|---|---|---|---|"]
        for metric in self._metrics:
            old = self._reference.get(metric) if self._reference else None
            new = self._candidate.get(metric) if self._candidate else None
            change = f"{new - old:+.4f}" if old is not None and new is not None else "n/a"
            rows.append(f"| {metric} | {self._format(old)} | {self._format(new)} | {change} |")
        return "\n".join(rows)

    @staticmethod
    def _format(value: float | None) -> str:
        return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"


class FactorHypothesisAgent(Agent[Hypothesis], HypothesisProposer):
    name = "factor_proposer"
    output_model = Hypothesis

    def __init__(
        self,
        llm: LLMClient,
        prompt: FactorProposerPrompt,
        tools: Sequence[Tool] = (),
        *,
        history_window: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, prompt, tools, **kwargs)
        self._history_window = history_window

    def propose(self, trace: Trace) -> Hypothesis:
        last = trace.records[-1].feedback if len(trace) else None
        return self._ask(
            history=trace.render_history(self._history_window),
            library=FactorLibrary(trace).render(),
            best=MetricComparison(
                trace.reference_result(), trace.baseline, labels=("baseline", "current best")
            ).render(),
            loops=len(trace),
            suggestion=last.next_hypothesis if last is not None else "",
            tools=self.tools,
        )


class FactorDesign(BaseModel):
    factors: list[FactorSpec] = Field(description="The factors that test the hypothesis, most important first")


class FactorDesignPolicy:
    """Checks a design: number of factors, unique names, no clash with existing features."""

    def __init__(self, max_factors: int, taken: Iterable[str]) -> None:
        self._max_factors = max_factors
        self._taken = {name.casefold() for name in taken}

    def __call__(self, design: FactorDesign) -> FactorDesign:
        names = [spec.name for spec in design.factors]
        if not names:
            raise OutputParsingError("Design at least one factor.")
        if len(names) > self._max_factors:
            raise OutputParsingError(f"Design at most {self._max_factors} factors, not {len(names)}.")
        if len({name.casefold() for name in names}) != len(names):
            raise OutputParsingError(f"Factor names must be unique: {names}")
        clashes = [name for name in names if name.casefold() in self._taken]
        if clashes:
            raise OutputParsingError(f"These names are already used by existing features, choose new ones: {clashes}")
        return design


class FactorDesignAgent(Agent[FactorDesign], ExperimentDesigner):
    name = "factor_designer"
    output_model = FactorDesign

    def __init__(
        self,
        llm: LLMClient,
        prompt: FactorDesignerPrompt,
        *,
        max_factors: int,
        reserved_names: Iterable[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, prompt, **kwargs)
        self._max_factors = max_factors
        self._reserved = set(reserved_names)

    def design(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        library = FactorLibrary(trace)
        policy = FactorDesignPolicy(self._max_factors, self._reserved | library.names())
        design = self._ask(
            policy,
            hypothesis=hypothesis.render(),
            library=library.render(),
            reserved=sorted(self._reserved),
            max_factors=self._max_factors,
        )
        return FactorExperiment(hypothesis=hypothesis, tasks=[FactorTask.from_spec(spec) for spec in design.factors])


class FactorCoderAgent(CoderAgent):
    name = "factor_coder"
    required_files = ("factor.py",)


class CodeReview(BaseModel):
    consistent: bool = Field(description="True when the code computes exactly the specified factor")
    score: float = Field(ge=0.0, le=1.0, description="Quality of the implementation, 1 = no remark")
    critique: str = Field(description="Concrete problems with line references, or a short confirmation")


class FactorCriticAgent(Agent[CodeReview]):
    name = "factor_critic"
    output_model = CodeReview

    def __init__(self, llm: LLMClient, prompt: FactorCriticPrompt, **kwargs: Any) -> None:
        super().__init__(llm, prompt, **kwargs)

    def review(self, task: Task, workspace: Workspace, value_summary: str) -> CodeReview:
        return self._ask(
            task=task.render(),
            code=workspace.render_code(),
            value_summary=value_summary,
        )


class FeedbackAnalysis(BaseModel):
    observations: str = Field(description="What the numbers show, compared with the current best")
    hypothesis_evaluation: str = Field(description="Whether and why the results support the hypothesis")
    hypothesis_supported: bool
    next_hypothesis: str = Field(description="The most promising hypothesis to test next, given this result")
    reason: str = Field(description="One-paragraph conclusion")


class FactorFeedbackAgent(Agent[FeedbackAnalysis], Summarizer):
    """Explains the result; the promotion decision itself is taken by the :class:`PromotionRule`."""

    name = "factor_analyst"
    output_model = FeedbackAnalysis

    def __init__(self, llm: LLMClient, prompt: FactorAnalystPrompt, rule: PromotionRule, **kwargs: Any) -> None:
        super().__init__(llm, prompt, **kwargs)
        self._rule = rule

    def summarize(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:
        if experiment.result is None:
            raise RunnerError("The experiment has no result to summarize")
        reference = trace.reference_result()
        verdict = self._rule.decide(experiment.result, reference)
        included = [
            task.name for task, impl in experiment.accepted() if impl.artifacts.get(FactorLibrary.INCLUDED) == "true"
        ]
        analysis = self._ask(
            hypothesis=experiment.hypothesis.render() if experiment.hypothesis else "(none)",
            tasks=experiment.render_tasks(),
            included=included,
            notes=experiment.result.notes,
            comparison=MetricComparison(experiment.result, reference).render(),
            verdict=verdict.reason,
            promoted=verdict.promote,
            history=trace.render_history(4),
        )
        return ExperimentFeedback(
            decision=verdict.promote,
            reason=f"{verdict.reason}. {analysis.reason}",
            observations=analysis.observations,
            hypothesis_evaluation=analysis.hypothesis_evaluation,
            hypothesis_supported=analysis.hypothesis_supported,
            next_hypothesis=analysis.next_hypothesis,
        )
