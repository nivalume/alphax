"""Run any function over a parameter grid and keep the parameters as labels.

A grid is a list of plain dicts, so it can be built here, by hand, or by
anything else::

    from alphax import params

    combos = params.grid(fast=[5, 10, 20], slow=[20, 50, 100], where=lambda p: p["fast"] < p["slow"])
    combos += params.random(20, seed=1, fast=params.IntUniform(2, 30), slow=params.IntUniform(30, 200))

    res = params.run(lambda fast, slow: backtest(close, fast, slow).sharpe_ratio()[0], combos)
    res.best()                      # the parameters with the highest value
    res.table()                     # {"fast": [...], "slow": [...], "value": [...]}
    params.neighbors(res, "value")  # how each point compares with the grid points around it

``run`` calls the function once per combination, in order, optionally on a
pool of threads or processes; the results come back in the grid's order
whatever the pool did, so a run is reproducible. A function returning a dict
gives one column per key; ``res.values`` keeps whatever it returned.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Choice",
    "IntUniform",
    "LogUniform",
    "ParamResults",
    "Uniform",
    "grid",
    "neighbors",
    "random",
    "run",
]


def grid(*, where: Callable[[dict], bool] | None = None, **values: Iterable[Any]) -> list[dict]:
    """Every combination of the values, first parameter outermost; with
    ``where``, only those it accepts (a conditional grid)."""
    if not values:
        raise ValueError("a grid needs at least one parameter")
    names = list(values)
    lists = []
    for name in names:
        v = values[name]
        v = list(v) if isinstance(v, Iterable) and not isinstance(v, (str, bytes)) else [v]
        if not v:
            raise ValueError(f"parameter {name!r} has no values")
        lists.append(v)
    out = [dict(zip(names, combo)) for combo in itertools.product(*lists)]
    return [p for p in out if where(p)] if where is not None else out


@dataclass(frozen=True)
class Choice:
    """One of ``options``, equally likely."""

    options: Sequence[Any]

    def draw(self, rng: np.random.Generator) -> Any:
        return self.options[int(rng.integers(len(self.options)))]


@dataclass(frozen=True)
class Uniform:
    """A float in ``[low, high)``."""

    low: float
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class IntUniform:
    """An integer in ``[low, high]``."""

    low: int
    high: int

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high + 1))


@dataclass(frozen=True)
class LogUniform:
    """A float whose logarithm is uniform on ``[log low, log high)``."""

    low: float
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))


def random(
    n: int,
    *,
    seed: int = 0,
    where: Callable[[dict], bool] | None = None,
    max_tries: int = 100,
    **dists: Any,
) -> list[dict]:
    """``n`` random combinations (random search). Each parameter is a
    distribution (:class:`Choice`, :class:`Uniform`, :class:`IntUniform`,
    :class:`LogUniform`), a list (a choice) or a constant. With ``where``,
    draws it rejects are redrawn, up to ``max_tries`` per combination.
    The same seed gives the same combinations."""
    if n < 0:
        raise ValueError("n cannot be negative")
    rng = np.random.default_rng(seed)
    spec = {}
    for name, d in dists.items():
        if hasattr(d, "draw"):
            spec[name] = d
        elif isinstance(d, (list, tuple)):
            spec[name] = Choice(list(d))
        else:
            spec[name] = Choice([d])
    out = []
    for _ in range(n):
        for _ in range(max_tries):
            p = {name: d.draw(rng) for name, d in spec.items()}
            if where is None or where(p):
                out.append(p)
                break
        else:
            raise ValueError(f"where rejected {max_tries} draws in a row")
    return out


class ParamResults:
    """What a function returned for each combination, with its parameters."""

    def __init__(self, combos: list[dict], values: list[Any]):
        self.params = combos
        self.values = values

    def __len__(self) -> int:
        return len(self.params)

    def __repr__(self) -> str:
        return f"ParamResults({len(self)} combinations of {self.names})"

    @property
    def names(self) -> list[str]:
        """Parameter names, in first-seen order."""
        seen: dict[str, None] = {}
        for p in self.params:
            seen.update(dict.fromkeys(p))
        return list(seen)

    def column(self, key: str = "value") -> np.ndarray:
        """One output per combination as floats: the value itself, or the
        ``key`` entry of a dict value."""
        out = []
        for v in self.values:
            if isinstance(v, Mapping):
                v = v.get(key, math.nan)
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(math.nan)
        return np.asarray(out)

    def table(self) -> dict[str, list]:
        """Columns: each parameter, then each output (``value``, or a dict
        value's keys)."""
        cols: dict[str, list] = {name: [p.get(name) for p in self.params] for name in self.names}
        keys: dict[str, None] = {}
        for v in self.values:
            keys.update(dict.fromkeys(v if isinstance(v, Mapping) else ["value"]))
        for key in keys:
            cols[key] = [v.get(key) if isinstance(v, Mapping) else v for v in self.values]
        return cols

    def to_frame(self) -> Any:
        """:meth:`table` as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.table())

    def best(self, key: str = "value", *, minimize: bool = False) -> dict:
        """The parameters with the highest (or lowest) output; ``NaN`` never
        wins, ties go to the first."""
        col = self.column(key)
        if np.isnan(col).all():
            raise ValueError(f"every {key!r} is NaN")
        i = int(np.nanargmin(col) if minimize else np.nanargmax(col))
        return dict(self.params[i])

    def select(self, **fixed: Any) -> ParamResults:
        """The combinations with these parameter values."""
        keep = [i for i, p in enumerate(self.params) if all(p.get(k) == v for k, v in fixed.items())]
        return ParamResults([self.params[i] for i in keep], [self.values[i] for i in keep])


def _call(args: tuple) -> Any:
    func, combo, extra, kwargs = args
    return func(*extra, **kwargs, **combo)


def run(
    func: Callable[..., Any],
    combos: Iterable[Mapping[str, Any]],
    *args: Any,
    n_jobs: int = 1,
    executor: str = "thread",
    **kwargs: Any,
) -> ParamResults:
    """``func(*args, **kwargs, **combo)`` for each combination.

    ``n_jobs > 1`` runs them on a pool: ``executor="thread"`` (alphax's
    backtests release the GIL) or ``"process"`` (``func`` and its arguments
    must pickle). Results are in ``combos`` order either way."""
    combos = [dict(c) for c in combos]
    if not combos:
        raise ValueError("no combinations to run")
    jobs = [(func, c, args, kwargs) for c in combos]
    if n_jobs <= 1:
        values = [_call(j) for j in jobs]
    else:
        pool = {"thread": ThreadPoolExecutor, "process": ProcessPoolExecutor}.get(executor)
        if pool is None:
            raise ValueError("executor must be 'thread' or 'process'")
        with pool(max_workers=n_jobs) as ex:
            values = list(ex.map(_call, jobs))
    return ParamResults(combos, values)


def neighbors(results: ParamResults, key: str = "value", names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Each combination against its grid neighbours: the combinations that
    differ in one parameter by one step of that parameter's sorted values.

    Returns arrays in ``results`` order: ``value``, ``mean`` and ``min`` of
    the neighbours' values, ``count`` of neighbours, and ``ratio`` =
    ``mean / value``. A peak with weak neighbours (a low ratio) is a fit to
    noise more often than a peak on a plateau. ``names`` limits which
    parameters count as axes (default: every parameter with orderable
    values)."""
    col = results.column(key)
    names = list(results.names if names is None else names)
    axes = {}
    for name in names:
        try:
            axes[name] = sorted({p[name] for p in results.params if name in p})
        except TypeError:
            continue
    index = {tuple(sorted(p.items(), key=lambda kv: kv[0])): i for i, p in enumerate(results.params)}
    out = {k: np.full(len(results), np.nan) for k in ("mean", "min", "ratio")}
    count = np.zeros(len(results), dtype=np.int64)
    for i, p in enumerate(results.params):
        found = []
        for name, values in axes.items():
            if name not in p:
                continue
            k = values.index(p[name])
            for step in (-1, 1):
                if 0 <= k + step < len(values):
                    q = dict(p, **{name: values[k + step]})
                    j = index.get(tuple(sorted(q.items(), key=lambda kv: kv[0])))
                    if j is not None and not np.isnan(col[j]):
                        found.append(col[j])
        count[i] = len(found)
        if found:
            out["mean"][i] = float(np.mean(found))
            out["min"][i] = float(np.min(found))
            with np.errstate(divide="ignore", invalid="ignore"):
                out["ratio"][i] = out["mean"][i] / col[i]
    return {"value": col, **out, "count": count}
