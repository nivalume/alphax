"""Analysis of return series that did not come from a alphax run: a fund's
daily returns, another engine's equity curve, a factor's long-short leg.

``returns`` is ``(bars,)`` or ``(bars, columns)`` of simple per-bar
returns; ``NaN`` is a bar without one. The metrics are the ones a
``Portfolio`` reports, computed by the same Rust code, so a portfolio's
``returns()`` gives the same numbers here as on the portfolio::

    alphax.returns.stats(r, benchmark=market, ann_factor=252)
    alphax.returns.rolling(r, "sharpe_ratio", window=63, ann_factor=252)
    alphax.returns.drawdowns(r)

``metrics()`` lists the metrics; those that compare against a benchmark
need one shaped like ``returns``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _alphax
from ._columns import columns, per_column, restore

__all__ = ["drawdowns", "metric", "metrics", "rolling", "stats", "value"]

_SOLO = (
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
    "sharpe_ratio",
    "downside_risk",
    "sortino_ratio",
    "tail_ratio",
    "value_at_risk",
    "cond_value_at_risk",
    "hit_ratio",
    "return_concentration",
)
_PAIRED = ("information_ratio", "beta", "alpha", "capture", "up_capture", "down_capture")


def metrics() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The metric names: those of a series alone, and those against a
    benchmark."""
    return _SOLO, _PAIRED


def _bench(benchmark: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if benchmark is None:
        return None
    b, _ = columns(benchmark, "benchmark")
    if b.shape[1] == 1 and shape[1] > 1:
        b = np.asfortranarray(np.repeat(b, shape[1], axis=1))
    return b


def metric(returns: Any, name: str, *, benchmark: Any = None, ann_factor: float = 365.0,
           risk_free: float = 0.0, required_return: float = 0.0, cutoff: float = 0.05, ddof: int = 1) -> Any:
    """One metric per column (a number for a 1-D series). A 1-D benchmark
    is every column's."""
    r, one_d = columns(returns, "returns")
    out = _alphax.ret_metric(r, name, _bench(benchmark, r.shape), float(ann_factor), float(risk_free),
                             float(required_return), float(cutoff), int(ddof))
    return per_column(out, one_d)


def rolling(returns: Any, name: str, window: int, *, min_periods: int | None = None, benchmark: Any = None,
            ann_factor: float = 365.0, risk_free: float = 0.0, required_return: float = 0.0,
            cutoff: float = 0.05, ddof: int = 1) -> np.ndarray:
    """A metric over each trailing ``window`` bars, shaped like ``returns``.
    ``min_periods`` (a full window by default) counts bars with a return."""
    r, one_d = columns(returns, "returns")
    out = _alphax.ret_rolling(r, name, int(window), None if min_periods is None else int(min_periods),
                              _bench(benchmark, r.shape), float(ann_factor), float(risk_free),
                              float(required_return), float(cutoff), int(ddof))
    return restore(out, one_d)


def value(returns: Any, start: float = 1.0) -> np.ndarray:
    """The value the returns compound to from ``start``; a ``NaN`` return
    leaves it where it was."""
    r, one_d = columns(returns, "returns")
    return restore(np.asfortranarray(start * np.cumprod(1.0 + np.nan_to_num(r), axis=0)), one_d)


def drawdowns(returns: Any) -> dict:
    """The drawdown episodes of the compounded value, as records: as
    ``Portfolio.drawdowns()`` gives them."""
    v, _ = columns(value(returns), "value")
    return _alphax.ret_drawdowns(v)


def stats(returns: Any, *, benchmark: Any = None, ann_factor: float = 365.0, risk_free: float = 0.0,
          required_return: float = 0.0, cutoff: float = 0.05) -> dict:
    """Every metric, and the drawdown count, the longest drawdown and the
    deepest drawdown's recovery, per column: the statistics table of a series
    with no trades behind it. With a benchmark, its metrics too.

    ``max_drawdown_recovery`` is the bars from the deepest drawdown's valley
    back to its peak, ``NaN`` while it has not recovered."""
    r, one_d = columns(returns, "returns")
    kw = dict(ann_factor=ann_factor, risk_free=risk_free, required_return=required_return, cutoff=cutoff)
    names = _SOLO + (_PAIRED if benchmark is not None else ())
    out = {name: metric(returns, name, benchmark=benchmark, **kw) for name in names}
    d = drawdowns(returns)
    n_cols = r.shape[1]
    count = np.bincount(d["col"], minlength=n_cols)
    longest = np.zeros(n_cols)
    recovery = np.full(n_cols, np.nan)
    for c in range(n_cols):
        mine = np.flatnonzero(d["col"] == c)
        dur = d["duration"][mine]
        longest[c] = dur.max() if len(dur) else 0.0
        if len(mine):
            deepest = mine[np.argmin(np.asarray(d["drawdown"])[mine])]
            if d["recovered"][deepest]:
                recovery[c] = float(d["recovery_duration"][deepest])
    out["drawdown_count"] = int(count[0]) if one_d else count
    out["max_drawdown_duration"] = float(longest[0]) if one_d else longest
    out["max_drawdown_recovery"] = float(recovery[0]) if one_d else recovery
    return out
