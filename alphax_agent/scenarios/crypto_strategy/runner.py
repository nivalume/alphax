"""Backtest of the accepted strategies on the full panel, with alphax."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from alphax_agent.core.exceptions import RunnerError
from alphax_agent.core.experiment import Experiment, ExperimentResult, Implementation
from alphax_agent.core.roles import Runner
from alphax_agent.core.task import Task
from alphax_agent.core.text import Excerpt
from alphax_agent.core.trace import Trace
from alphax_agent.env.alphax import AlphaXEnvironment
from alphax_agent.scenarios.crypto_strategy.agents import SELECTED
from alphax_agent.scenarios.crypto_strategy.data import StrategyDataCatalog
from alphax_agent.scenarios.crypto_strategy.settings import CryptoStrategySettings
from alphax_agent.scenarios.crypto_strategy.workspace import StrategyWorkspace

#: Artifacts of an implementation and of the result.
METRICS = "metrics"
OOS_METRICS = "oos_metrics"


def flatten(segment: str, metrics: dict[str, Any]) -> dict[str, float]:
    """``{"portfolio.sharpe": 1.2}`` of a segment as ``{"is.portfolio.sharpe": 1.2}``; missing values are dropped."""
    return {
        f"{segment}.{name}": float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    }


class AlphaXStrategyRunner(Runner):
    """Backtests every accepted strategy of an experiment on the full panel; the best one is the result.

    Only the in-sample metrics enter the result, which the agents see and the promotion rule judges. The
    out-of-sample metrics are written next to the backtest (``oos_metrics.json``) and named in the result's
    artifacts, for whoever reads the run.
    """

    def __init__(
        self,
        environment: AlphaXEnvironment,
        settings: CryptoStrategySettings,
        catalog: StrategyDataCatalog,
        workspace_root: Path,
    ) -> None:
        self._env = environment
        self._settings = settings
        self._catalog = catalog
        self._root = Path(workspace_root)

    def backtest_config(self) -> dict[str, Any]:
        settings = self._settings
        backtest = settings.backtest
        period = settings.period
        return {
            "signal_delay": backtest.signal_delay,
            "fees": backtest.fees,
            "slippage": backtest.slippage,
            "init_cash": backtest.init_cash,
            "ann_factor": settings.alphax.ann_factor,
            "threads": settings.alphax.threads,
            "start_ns": _nanoseconds(period.start.isoformat()),
            "oos_start_ns": _nanoseconds(period.oos_start.isoformat()),
            "end_ns": _nanoseconds(period.end.isoformat()),
        }

    def run(self, experiment: Experiment, trace: Trace) -> Experiment:
        accepted = experiment.accepted()
        if not accepted:
            raise RunnerError("The experiment has no accepted implementation to run")
        notes: list[str] = []
        results: dict[str, ExperimentResult] = {}
        for task, implementation in accepted:
            result = self._backtest(experiment, task, implementation, notes)
            if result is not None:
                results[task.name] = result
        if not results:
            raise RunnerError("No strategy could be backtested:\n" + "\n".join(notes))
        primary = self._settings.promotion.primary
        best = max(results, key=lambda name: self._rank(results[name].get(primary.metric), primary.higher_is_better))
        for name, result in results.items():
            summary = ", ".join(
                f"{metric}={result.get(metric):.4f}"
                for metric in ("is.portfolio.sharpe", "is.portfolio.annualized_return", "is.trade_count")
                if result.get(metric) is not None
            )
            notes.append(f"{name}: {summary}" + (" (selected)" if name == best else ""))
        chosen = results[best]
        experiment.workspace = StrategyWorkspace(
            path=Path(chosen.artifacts["workspace"]), data_file=self._catalog.path("full")
        )
        experiment.result = chosen.model_copy(update={"notes": notes})
        return experiment

    def _backtest(
        self, experiment: Experiment, task: Task, implementation: Implementation, notes: list[str]
    ) -> ExperimentResult | None:
        workspace = StrategyWorkspace(
            path=self._root / experiment.id / task.name,
            files=dict(implementation.workspace.files),
            data_file=self._catalog.path("full"),
        )
        execution = workspace.backtest(self._env, self.backtest_config(), self._settings.alphax.timeout)
        if not execution.ok or execution.metrics is None:
            detail = execution.read_error or Excerpt.tail(execution.result.output, 1500)
            notes.append(f"{task.name} failed in the backtest: {detail}")
            implementation.artifacts["backtest_error"] = detail
            return None
        in_sample = flatten("is", execution.metrics.get("is", {}))
        if not in_sample:
            notes.append(f"{task.name}: the backtest has no in-sample bar")
            return None
        oos_path = workspace.resolve("oos_metrics.json")
        oos_path.write_text(json.dumps(execution.metrics.get("oos", {}), indent=2), encoding="utf-8")
        metrics_path = workspace.resolve(StrategyWorkspace.METRICS)
        implementation.artifacts[METRICS] = str(metrics_path)
        implementation.artifacts[OOS_METRICS] = str(oos_path)
        return ExperimentResult(
            metrics=in_sample,
            artifacts={
                "workspace": str(workspace.path),
                SELECTED: task.name,
                METRICS: str(metrics_path),
                OOS_METRICS: str(oos_path),
            },
            log=Excerpt.tail(execution.result.output, 4000),
        )

    @staticmethod
    def _rank(value: float | None, higher_is_better: bool) -> float:
        if value is None or not math.isfinite(value):
            return -math.inf
        return value if higher_is_better else -value


def _nanoseconds(day: str) -> int:
    return int(np.datetime64(day, "ns").astype(np.int64))
