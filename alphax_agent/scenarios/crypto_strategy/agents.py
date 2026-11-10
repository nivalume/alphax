"""The two LLM-backed roles of the crypto strategy scenario: the hypothesis and the experiment design.

A strategy here is a tradeable rule on one instrument's own bars (and funding), justified by an economic
mechanism. The proposer states the mechanism; the designer turns it into the rules a backtest could run.
What a strategy has to contain is asked for in the prompts, not fixed in a schema: the designer returns a
name and a free-form description per strategy, and the experiment is the core :class:`Experiment`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, field_validator

from alphax_agent.agents.agent import Agent
from alphax_agent.core.exceptions import OutputParsingError
from alphax_agent.core.experiment import Experiment, Hypothesis
from alphax_agent.core.roles import ExperimentDesigner, HypothesisProposer
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace
from alphax_agent.llm.client import LLMClient
from alphax_agent.scenarios.crypto_strategy.prompts import StrategyDesignerPrompt, StrategyProposerPrompt


class StrategyHypothesisAgent(Agent[Hypothesis], HypothesisProposer):
    """Proposes one economic mechanism, and the rule it implies, per loop."""

    name = "strategy_proposer"
    output_model = Hypothesis

    SIMPLE_FIRST_LOOPS = 15
    """Until this many loops have run, the proposer is told to start from the simplest mechanism."""

    def __init__(
        self,
        llm: LLMClient,
        prompt: StrategyProposerPrompt,
        *,
        history_window: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, prompt, **kwargs)
        self._history_window = history_window

    def propose(self, trace: Trace) -> Hypothesis:
        last = trace.records[-1].feedback if len(trace) else None
        return self._ask(
            history=trace.render_history(self._history_window),
            loops=len(trace),
            simple_first=len(trace) < self.SIMPLE_FIRST_LOOPS,
            suggestion=last.next_hypothesis if last is not None else "",
        )


class StrategyDraft(BaseModel):
    """One strategy as the designer writes it: a name and the whole write-up."""

    name: str = Field(description="Identifier of the strategy: letters, digits and underscores, starting with a letter")
    description: str = Field(
        description=(
            "The complete, self-contained write-up of the strategy: its phenomenon, mechanism, return source, "
            "risk borne, the LaTeX of the signal, the position and the gate, the horizon in bars, the testable "
            "prediction, when it fails, and every parameter with its meaning"
        )
    )

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
            raise ValueError("strategy names must be identifiers of at most 64 characters starting with a letter")
        return value

    @field_validator("description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a strategy needs a description")
        return value


class ExperimentDraft(BaseModel):
    strategies: list[StrategyDraft] = Field(
        description="The strategies that test the hypothesis, most important first: one mechanism and its variants"
    )


class StrategyDesignPolicy:
    """Checks a design: how many strategies, unique names, none proposed in an earlier loop.

    Only the bookkeeping is checked. What a strategy says is left to the prompt and, later, to the backtest.
    """

    def __init__(self, max_strategies: int, taken: Iterable[str]) -> None:
        self._max_strategies = max_strategies
        self._taken = {name.casefold() for name in taken}

    def __call__(self, design: ExperimentDraft) -> ExperimentDraft:
        names = [strategy.name for strategy in design.strategies]
        if not names:
            raise OutputParsingError("Design at least one strategy.")
        if len(names) > self._max_strategies:
            raise OutputParsingError(f"Design at most {self._max_strategies} strategies, not {len(names)}.")
        if len({name.casefold() for name in names}) != len(names):
            raise OutputParsingError(f"Strategy names must be unique: {names}")
        clashes = [name for name in names if name.casefold() in self._taken]
        if clashes:
            raise OutputParsingError(
                f"These names were used by strategies of earlier loops, which are not to be re-proposed; "
                f"choose new ones: {clashes}"
            )
        return design


class StrategyDesignAgent(Agent[ExperimentDraft], ExperimentDesigner):
    """Turns the mechanism into the rules a backtest can run."""

    name = "strategy_designer"
    output_model = ExperimentDraft

    def __init__(
        self,
        llm: LLMClient,
        prompt: StrategyDesignerPrompt,
        *,
        max_strategies: int,
        history_window: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, prompt, **kwargs)
        self._max_strategies = max_strategies
        self._history_window = history_window

    def design(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        taken = sorted(
            {task.name for record in trace.records if record.experiment is not None for task in record.experiment.tasks}
        )
        draft = self._ask(
            StrategyDesignPolicy(self._max_strategies, taken),
            hypothesis=hypothesis.render(),
            history=trace.render_history(self._history_window),
            taken=taken,
            max_strategies=self._max_strategies,
        )
        return Experiment(
            hypothesis=hypothesis,
            tasks=[Task(name=strategy.name, description=strategy.description) for strategy in draft.strategies],
        )
