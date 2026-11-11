"""Evaluation stages of a strategy implementation.

execution -> format -> look-ahead -> critic. The first three are deterministic and cost no tokens; the
look-ahead stage reruns the code on a truncated panel and rejects any strategy whose earlier positions change
when later bars are removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphax_agent.coder.evaluation import EvaluationContext, EvaluationStage
from alphax_agent.core.feedback import StageResult
from alphax_agent.core.text import Excerpt
from alphax_agent.env.base import Environment
from alphax_agent.scenarios.crypto_strategy.data import StrategyDataCatalog
from alphax_agent.scenarios.crypto_strategy.workspace import StrategyWorkspace

if TYPE_CHECKING:
    from alphax_agent.scenarios.crypto_strategy.agents import StrategyCriticAgent

#: Positions are fractions of equity; this much outside [-1, 1] is a rounding error, not a breach.
_BOUND_TOLERANCE = 1e-9


def summarize_positions(positions: np.ndarray, close: np.ndarray) -> str:
    """Distribution of the positions on the bars an instrument traded, for the critic and the feedback."""
    if positions.shape != close.shape or not np.issubdtype(positions.dtype, np.number):
        return f"shape {positions.shape}, dtype {positions.dtype}"
    weights = np.where(np.isfinite(positions), positions, 0.0)
    listed = np.isfinite(close)
    values = weights[listed]
    if values.size == 0:
        return f"shape {positions.shape}; no traded bar"
    change = np.abs(np.diff(weights, axis=0)).mean() if len(weights) > 1 else 0.0
    return (
        f"shape {positions.shape}; on traded bars: long {np.mean(values > 0):.1%}, short {np.mean(values < 0):.1%}, "
        f"flat {np.mean(values == 0):.1%}; mean |position| {np.abs(values).mean():.4f}, "
        f"min {values.min():.4f}, max {values.max():.4f}; mean |change| per bar {change:.4f}"
    )


class StrategyExecutionStage(EvaluationStage):
    name = "execution"

    def __init__(self, environment: Environment, timeout: float, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._env = environment
        self._timeout = timeout

    def evaluate(self, context: EvaluationContext) -> StageResult:
        workspace = context.workspace
        if not isinstance(workspace, StrategyWorkspace):
            raise TypeError("StrategyExecutionStage needs a StrategyWorkspace")
        execution = workspace.execute(self._env, self._timeout)
        output = Excerpt.tail(execution.result.output.strip(), 3000) or "(no output)"
        if not execution.result.ok:
            reason = "timed out" if execution.result.timed_out else f"exited with code {execution.result.exit_code}"
            return self.result(False, 0.0, f"Running positions() {reason}.\nOutput:\n{output}")
        if execution.positions is None:
            return self.result(False, 0.2, f"The script ran but {execution.read_error}.\nOutput:\n{output}")
        context.put("positions", execution.positions)
        return self.result(True, 1.0, f"Ran in {execution.result.seconds:.1f}s.\nOutput:\n{output}")


class StrategyFormatStage(EvaluationStage):
    """Checks that the positions are bounded, finite and actually trade."""

    name = "format"

    def __init__(self, catalog: StrategyDataCatalog, min_active_share: float, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._catalog = catalog
        self._min_active_share = min_active_share

    def evaluate(self, context: EvaluationContext) -> StageResult:
        positions: np.ndarray = context.get("positions")
        close = self._catalog.panel("debug")["close"]
        checks = self.checks(positions, close)
        passed = [ok for _, ok, _ in checks]
        lines = [f"- {'OK  ' if ok else 'FAIL'} {name}: {detail}" for name, ok, detail in checks]
        summary = summarize_positions(positions, close)
        context.put("position_summary", summary)
        message = "\n".join(lines) + f"\n\nPosition summary:\n{summary}"
        return self.result(all(passed), sum(passed) / len(passed), message)

    def checks(self, positions: np.ndarray, close: np.ndarray) -> list[tuple[str, bool, str]]:
        checks: list[tuple[str, bool, str]] = [
            ("shape", positions.shape == close.shape, f"shape is {positions.shape}, expected {close.shape}")
        ]
        numeric = np.issubdtype(positions.dtype, np.number)
        checks.append(("dtype", bool(numeric), f"dtype is {positions.dtype}"))
        if not checks[0][1] or not numeric:
            return checks
        values = positions.astype("float64")
        listed = np.isfinite(close)
        infinite = int(np.isinf(values).sum())
        weights = np.where(np.isfinite(values), values, 0.0)
        outside = int((np.abs(weights) > 1.0 + _BOUND_TOLERANCE).sum())
        active = float((np.abs(weights[listed]) > 0).mean()) if listed.any() else 0.0
        varying = bool(np.any(np.abs(np.diff(np.where(listed, weights, 0.0), axis=0)) > 0))
        checks += [
            ("finite", infinite == 0, f"{infinite} infinite values (use NaN or 0 for flat)"),
            ("bounded", outside == 0, f"{outside} positions outside [-1, 1]"),
            (
                "active",
                active >= self._min_active_share,
                f"a position is held on {active:.1%} of the traded bars, at least {self._min_active_share:.0%} needed",
            ),
            ("varying", varying, "the positions change over time" if varying else "the positions never change"),
        ]
        return checks


class StrategyLookAheadStage(EvaluationStage):
    """Rejects strategies whose position at a bar depends on later bars."""

    name = "look_ahead"

    def __init__(
        self,
        environment: Environment,
        catalog: StrategyDataCatalog,
        timeout: float,
        *,
        max_mismatch: float = 0.001,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        self._env = environment
        self._catalog = catalog
        self._timeout = timeout
        self._max_mismatch = max_mismatch

    def evaluate(self, context: EvaluationContext) -> StageResult:
        workspace = context.workspace
        if not isinstance(workspace, StrategyWorkspace):
            raise TypeError("StrategyLookAheadStage needs a StrategyWorkspace")
        positions: np.ndarray = context.get("positions")
        cut = self._catalog.cut_index
        cut_date = self._catalog.meta["cut_date"]
        truncated = workspace.with_data(self._catalog.path("debug_cut"), workspace.path / "_look_ahead")
        execution = truncated.execute(self._env, self._timeout)
        if not execution.ok or execution.positions is None:
            detail = execution.read_error or Excerpt.tail(execution.result.output, 2000)
            return self.result(False, 0.0, f"The code fails when the data ends on {cut_date}:\n{detail}")
        full = positions[:cut].astype("float64")
        short = execution.positions.astype("float64")
        if short.shape != full.shape:
            return self.result(
                False,
                0.0,
                f"On the panel cut at {cut_date} the positions have shape {short.shape}, expected {full.shape}",
            )
        both_nan = np.isnan(full) & np.isnan(short)
        differs = ~(np.isclose(full, short, rtol=1e-6, atol=1e-9) | both_nan)
        ratio = float(differs.mean()) if differs.size else 0.0
        if ratio <= self._max_mismatch:
            return self.result(True, 1.0, f"Positions up to {cut_date} are unchanged when later bars are removed.")
        row, column = np.argwhere(differs)[0]
        return self.result(
            False,
            0.0,
            f"{ratio:.2%} of the positions up to {cut_date} change when the bars after it are removed (e.g. row "
            f"{row}, instrument {column}: {full[row, column]!r} with the full panel vs {short[row, column]!r} cut). "
            "The position at row t must only use rows up to t: no negative shift, no centered window, no statistic "
            "over the whole panel (global mean, std, quantile or rank), no backward fill.",
        )


class StrategyCriticStage(EvaluationStage):
    """An LLM review of whether the code implements the strategy that was specified."""

    name = "critic"

    def __init__(self, critic: StrategyCriticAgent, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._critic = critic

    def evaluate(self, context: EvaluationContext) -> StageResult:
        review = self._critic.review(context.task, context.workspace, context.get("position_summary"))
        return self.result(review.consistent, review.score, review.critique)
