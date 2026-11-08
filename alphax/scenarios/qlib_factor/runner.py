"""Backtest of the factor library plus the new factors in qlib."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined

from alphax.core.exceptions import ExecutionError, RunnerError
from alphax.core.experiment import Experiment, ExperimentResult, Implementation
from alphax.core.roles import Runner
from alphax.core.task import Task
from alphax.core.text import Excerpt
from alphax.core.trace import Trace
from alphax.core.workspace import Workspace
from alphax.env.qlib import QlibEnvironment
from alphax.scenarios.qlib_factor.data import FactorDataCatalog
from alphax.scenarios.qlib_factor.library import FactorLibrary
from alphax.scenarios.qlib_factor.settings import QlibFactorSettings
from alphax.scenarios.qlib_factor.workspace import FactorWorkspace

logger = logging.getLogger(__name__)


class QlibWorkflowConfig:
    """Renders the qlib workflow file of a backtest."""

    TEMPLATE = "templates/workflow.yaml.j2"
    FILE_NAME = "workflow.yaml"
    FACTORS_FILE = "combined_factors.parquet"

    def __init__(self, settings: QlibFactorSettings, calendar: Sequence[str] = ()) -> None:
        self._settings = settings
        self._backtest_end = self._last_tradable_date(settings.segments.test[1], calendar)
        source = resources.files("alphax.scenarios.qlib_factor").joinpath(self.TEMPLATE).read_text(encoding="utf-8")
        self._template = JinjaEnvironment(
            undefined=StrictUndefined, trim_blocks=True, keep_trailing_newline=True
        ).from_string(source)

    def render(self, *, with_factors: bool) -> str:
        settings = self._settings
        backtest = settings.backtest
        exchange = {
            "deal_price": backtest.deal_price,
            "open_cost": backtest.open_cost,
            "close_cost": backtest.close_cost,
            "min_cost": backtest.min_cost,
        }
        if backtest.limit_threshold is not None:
            exchange["limit_threshold"] = backtest.limit_threshold
        return self._template.render(
            provider_uri=str(settings.qlib.provider_path),
            region=settings.qlib.region,
            market=settings.market,
            benchmark=settings.benchmark,
            segments=settings.segments,
            label=settings.label,
            feature_expressions=list(settings.base_features.values()),
            feature_names=list(settings.base_features),
            factors_file=self.FACTORS_FILE if with_factors else None,
            backtest=backtest,
            backtest_end=self._backtest_end,
            exchange=exchange,
            model=settings.model,
            ann_scaler=settings.ann_scaler,
        )

    @staticmethod
    def _last_tradable_date(end: str, calendar: Sequence[str]) -> str:
        """qlib trades on the day after each decision, so the backtest must end before the last trading day."""
        tradable = [day for day in calendar[:-1] if day <= end]
        return tradable[-1] if tradable else end


class FactorCombiner:
    """Drops new factors that duplicate kept ones and assembles the model's factor table."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def deduplicate(self, candidates: pd.DataFrame, kept: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
        notes: list[str] = []
        selected: list[str] = []
        pool = {str(name): kept[name] for name in kept.columns} if kept is not None else {}
        for name in candidates.columns:
            series = candidates[name]
            similar = [(other, self.cross_sectional_correlation(series, values)) for other, values in pool.items()]
            match = max(similar, key=lambda item: abs(item[1]), default=None)
            if match is not None and abs(match[1]) >= self._threshold:
                notes.append(f"{name} was dropped: mean cross-sectional correlation {match[1]:.4f} with {match[0]}")
                continue
            selected.append(str(name))
            pool[str(name)] = series
        return candidates[selected], notes

    def combine(self, frames: Iterable[pd.DataFrame | None]) -> pd.DataFrame:
        present = [frame for frame in frames if frame is not None and not frame.empty]
        if not present:
            raise RunnerError("There is no factor to combine")
        combined = pd.concat(present, axis=1, join="outer").sort_index()
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
        combined.columns = pd.MultiIndex.from_product([["feature"], [str(name) for name in combined.columns]])
        return combined.astype("float64")

    @staticmethod
    def cross_sectional_correlation(left: pd.Series, right: pd.Series) -> float:
        """Mean over dates of the Pearson correlation across instruments."""
        frame = pd.concat([left.rename("a"), right.rename("b")], axis=1, join="inner")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            return 0.0
        demeaned = frame - frame.groupby(level="datetime").transform("mean")
        sums = (
            pd.DataFrame(
                {
                    "ab": demeaned["a"] * demeaned["b"],
                    "aa": demeaned["a"] ** 2,
                    "bb": demeaned["b"] ** 2,
                }
            )
            .groupby(level="datetime")
            .sum()
        )
        covariance, scale = sums["ab"], np.sqrt(sums["aa"] * sums["bb"])
        correlation = (covariance / scale).replace([np.inf, -np.inf], np.nan).dropna()
        return float(correlation.mean()) if len(correlation) else 0.0


class QlibFactorRunner(Runner):
    """Computes accepted factors on the full sample and backtests them with the library."""

    def __init__(
        self,
        environment: QlibEnvironment,
        settings: QlibFactorSettings,
        catalog: FactorDataCatalog,
        workspace_root: Path,
        *,
        combiner: FactorCombiner | None = None,
        config: QlibWorkflowConfig | None = None,
    ) -> None:
        self._env = environment
        self._settings = settings
        self._catalog = catalog
        self._root = Path(workspace_root)
        self._combiner = combiner or FactorCombiner(settings.correlation_threshold)
        self._config = config or QlibWorkflowConfig(settings, environment.trading_calendar())

    def establish_baseline(self) -> ExperimentResult:
        notes = [f"base features only ({len(self._settings.base_features)})"]
        return self._backtest(Workspace(path=self._root / "baseline"), None, notes)

    def run(self, experiment: Experiment, trace: Trace) -> Experiment:
        accepted = experiment.accepted()
        if not accepted:
            raise RunnerError("The experiment has no accepted implementation to run")
        notes: list[str] = []
        candidates = {
            task.name: values
            for task, implementation in accepted
            if (values := self._full_sample_values(task, implementation, notes)) is not None
        }
        if not candidates:
            raise RunnerError("No factor could be computed on the full sample:\n" + "\n".join(notes))
        library = FactorLibrary(trace)
        library_values = library.load_values()
        kept, duplicates = self._combiner.deduplicate(pd.DataFrame(candidates), library_values)
        notes += duplicates
        for task, implementation in accepted:
            implementation.artifacts[FactorLibrary.INCLUDED] = "true" if task.name in kept.columns else "false"
        if kept.empty:
            raise RunnerError("Every new factor duplicates an existing one:\n" + "\n".join(notes))
        library_size = 0 if library_values is None else library_values.shape[1]
        notes.append(
            f"model features: {len(self._settings.base_features)} base, {library_size} from the library, "
            f"{kept.shape[1]} new ({', '.join(map(str, kept.columns))})"
        )
        workspace = Workspace(path=self._root / experiment.id)
        experiment.workspace = workspace
        experiment.result = self._backtest(workspace, self._combiner.combine([library_values, kept]), notes)
        return experiment

    def _full_sample_values(self, task: Task, implementation: Implementation, notes: list[str]) -> pd.Series | None:
        source = implementation.workspace
        workspace = FactorWorkspace(
            path=source.path.with_name(f"{source.path.name}_full"),
            files=dict(source.files),
            data_file=self._catalog.path("full"),
        )
        execution = workspace.execute(self._env, self._settings.factor_timeout)
        if not execution.ok:
            detail = execution.read_error or Excerpt.tail(execution.result.output, 1500)
            notes.append(f"{task.name} failed on the full sample: {detail}")
            implementation.artifacts["full_sample_error"] = detail
            return None
        series = execution.first_column().astype("float64").replace([np.inf, -np.inf], np.nan).rename(task.name)
        path = workspace.resolve(FactorWorkspace.VALUES)
        series.to_frame().to_parquet(path)
        implementation.artifacts[FactorLibrary.VALUES] = str(path)
        return series

    def _backtest(self, workspace: Workspace, factors: pd.DataFrame | None, notes: list[str]) -> ExperimentResult:
        workspace.materialize()
        if factors is not None:
            factors.to_parquet(workspace.resolve(QlibWorkflowConfig.FACTORS_FILE))
        workspace.inject({QlibWorkflowConfig.FILE_NAME: self._config.render(with_factors=factors is not None)})
        try:
            outcome = self._env.run_workflow(workspace.path, QlibWorkflowConfig.FILE_NAME)
        except ExecutionError as exc:
            raise RunnerError(str(exc)) from exc
        artifacts = {"workspace": str(workspace.path)}
        if outcome.report_path:
            artifacts["portfolio_report"] = outcome.report_path
        return ExperimentResult(metrics=outcome.metrics, artifacts=artifacts, notes=notes, log=outcome.log)
