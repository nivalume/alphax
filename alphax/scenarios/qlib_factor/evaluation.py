"""Evaluation stages of a factor implementation.

execution → format → look-ahead → critic. The first three are deterministic and cost no
tokens; the look-ahead stage reruns the code on a truncated history and rejects any factor
whose past values change when future data is removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphax.coder.evaluation import EvaluationContext, EvaluationStage
from alphax.core.feedback import StageResult
from alphax.core.text import Excerpt
from alphax.env.base import Environment
from alphax.scenarios.qlib_factor.data import FactorDataCatalog
from alphax.scenarios.qlib_factor.workspace import FactorWorkspace

if TYPE_CHECKING:
    from alphax.scenarios.qlib_factor.agents import FactorCriticAgent


class FactorExecutionStage(EvaluationStage):
    name = "execution"

    def __init__(self, environment: Environment, timeout: float, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._env = environment
        self._timeout = timeout

    def evaluate(self, context: EvaluationContext) -> StageResult:
        workspace = context.workspace
        if not isinstance(workspace, FactorWorkspace):
            raise TypeError("FactorExecutionStage needs a FactorWorkspace")
        execution = workspace.execute(self._env, self._timeout)
        output = Excerpt.tail(execution.result.output.strip(), 3000) or "(no output)"
        if not execution.result.ok:
            reason = "timed out" if execution.result.timed_out else f"exited with code {execution.result.exit_code}"
            return self.result(False, 0.0, f"`python factor.py` {reason}.\nOutput:\n{output}")
        if execution.values is None:
            return self.result(False, 0.2, f"The script ran but {execution.read_error}.\nOutput:\n{output}")
        context.put("values", execution.values)
        return self.result(True, 1.0, f"Ran in {execution.result.seconds:.1f}s.\nOutput:\n{output}")


class FactorFormatStage(EvaluationStage):
    """Checks the shape, index, dtype and quality of the factor values."""

    name = "format"

    def __init__(self, catalog: FactorDataCatalog, max_nan_ratio: float, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._catalog = catalog
        self._max_nan_ratio = max_nan_ratio

    def evaluate(self, context: EvaluationContext) -> StageResult:
        values: pd.DataFrame = context.get("values")
        source = self._catalog.frame("debug")
        checks = self._checks(values, source, context.task.name)
        passed = [ok for _, ok, _ in checks]
        lines = [f"- {'OK  ' if ok else 'FAIL'} {name}: {detail}" for name, ok, detail in checks]
        summary = self._summary(values)
        context.put("value_summary", summary)
        message = "\n".join(lines) + f"\n\nValue summary:\n{summary}"
        return self.result(all(passed), sum(passed) / len(passed), message)

    def _checks(self, values: pd.DataFrame, source: pd.DataFrame, factor_name: str) -> list[tuple[str, bool, str]]:
        index_names = list(values.index.names)
        columns = [str(column) for column in values.columns]
        checks: list[tuple[str, bool, str]] = [
            (
                "index",
                index_names == ["datetime", "instrument"],
                f"index levels are {index_names}, expected ['datetime', 'instrument']",
            ),
            ("columns", columns == [factor_name], f"columns are {columns}, expected exactly ['{factor_name}']"),
        ]
        if len(values.columns) != 1:
            return checks
        series = values.iloc[:, 0]
        numeric = pd.api.types.is_numeric_dtype(series)
        checks.append(("dtype", numeric, f"dtype is {series.dtype}"))
        if not numeric or index_names != ["datetime", "instrument"]:
            return checks
        finite = series.replace([np.inf, -np.inf], np.nan)
        infinite = int(np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).sum())
        nan_ratio = float(finite.isna().mean()) if len(finite) else 1.0
        duplicated = int(values.index.duplicated().sum())
        coverage = float(values.index.isin(source.index).mean()) if len(values) else 0.0
        reach = float(source.index.isin(values.index).mean()) if len(source) else 0.0
        dispersion = finite.groupby(level="datetime").std()
        varying = float((dispersion > 1e-12).mean()) if len(dispersion) else 0.0
        checks += [
            ("not_empty", len(values) > 0, f"{len(values)} rows"),
            ("unique_index", duplicated == 0, f"{duplicated} duplicated (datetime, instrument) pairs"),
            ("aligned", coverage >= 0.99, f"{coverage:.1%} of the output rows exist in the input data"),
            ("coverage", reach >= 0.5, f"{reach:.1%} of the input rows have a factor value"),
            ("finite", infinite == 0, f"{infinite} infinite values"),
            ("nan_ratio", nan_ratio <= self._max_nan_ratio, f"{nan_ratio:.1%} NaN, limit {self._max_nan_ratio:.0%}"),
            ("cross_section", varying >= 0.5, f"values differ across instruments on {varying:.1%} of the dates"),
        ]
        return checks

    @staticmethod
    def _summary(values: pd.DataFrame) -> str:
        if values.shape[1] != 1 or not pd.api.types.is_numeric_dtype(values.iloc[:, 0]):
            return f"shape {values.shape}, dtypes {dict(values.dtypes.astype(str))}"
        series = values.iloc[:, 0].replace([np.inf, -np.inf], np.nan)
        stats = series.describe(percentiles=[0.01, 0.5, 0.99])
        return f"shape {values.shape}; " + ", ".join(f"{key}={value:.6g}" for key, value in stats.items())


class LookAheadStage(EvaluationStage):
    """Rejects factors that use information from after the date they are stamped with."""

    name = "look_ahead"

    def __init__(
        self,
        environment: Environment,
        catalog: FactorDataCatalog,
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
        if not isinstance(workspace, FactorWorkspace):
            raise TypeError("LookAheadStage needs a FactorWorkspace")
        values: pd.DataFrame = context.get("values")
        cut = self._catalog.cut_date
        truncated = workspace.with_data(self._catalog.path("debug_cut"), workspace.path / "_look_ahead")
        execution = truncated.execute(self._env, self._timeout)
        if not execution.ok:
            detail = execution.read_error or Excerpt.tail(execution.result.output, 2000)
            return self.result(False, 0.0, f"The code fails when the history ends on {cut.date()}:\n{detail}")
        full = values.iloc[:, 0]
        full = full[full.index.get_level_values("datetime") <= cut]
        short = execution.first_column()
        left, right = full.align(short, join="outer")
        both_nan = left.isna() & right.isna()
        close = np.isclose(
            left.to_numpy(dtype="float64", na_value=np.nan),
            right.to_numpy(dtype="float64", na_value=np.nan),
            rtol=1e-6,
            atol=1e-9,
        )
        differs = ~(close | both_nan.to_numpy())
        ratio = float(differs.mean()) if len(differs) else 0.0
        if ratio <= self._max_mismatch:
            return self.result(True, 1.0, f"Values up to {cut.date()} are unchanged when later data is removed.")
        position = int(differs.argmax())
        example = left.index[position]
        return self.result(
            False,
            0.0,
            f"{ratio:.2%} of the values up to {cut.date()} change when the data after {cut.date()} is removed "
            f"(e.g. {example}: {left.iloc[position]!r} with the full history vs {right.iloc[position]!r} truncated). "
            "The value at date t must only use data up to t: no negative shift, no centered window, "
            "no normalisation over the whole sample, no backward fill.",
        )


class FactorCriticStage(EvaluationStage):
    """An LLM review of whether the code computes the factor that was specified."""

    name = "critic"

    def __init__(self, critic: FactorCriticAgent, weight: float = 1.0) -> None:
        super().__init__(weight)
        self._critic = critic

    def evaluate(self, context: EvaluationContext) -> StageResult:
        review = self._critic.review(context.task, context.workspace, context.get("value_summary"))
        return self.result(review.consistent, review.score, review.critique)
