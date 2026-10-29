"""Charts of a portfolio, with Plotly. Needs ``pip install phasor-trader[plot]``.

Each function returns a ``plotly.graph_objects.Figure`` to show, combine or
save; ``index`` puts the bars' times on the x axis::

    phasor.plotting.value(pf, 0, index=dates).show()
    phasor.plotting.orders(pf, 0, index=dates)
    phasor.plotting.drawdowns(pf, 0)
    phasor.plotting.rolling(pf, "sharpe_ratio", 63)

Factor reports have their own charts: ``factor_ic``, ``factor_decay``,
``factor_quantiles`` and ``factor_turnover``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "drawdowns",
    "factor_decay",
    "factor_ic",
    "factor_quantiles",
    "factor_turnover",
    "orders",
    "rolling",
    "value",
]

_UP, _DOWN, _LINE, _MUTED = "#2a9d8f", "#d1495b", "#3d5a80", "#98a6b3"
_QUANTILES = ("#3d5a80", "#577590", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#d1495b")


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("phasor.plotting needs plotly: pip install phasor-trader[plot]") from e
    return go


def _x(n: int, index: Any) -> np.ndarray:
    return np.arange(n) if index is None else np.asarray(index)


def _axis(n: int, index: Any, report: Any) -> np.ndarray:
    """``index`` when given, otherwise the report's index, otherwise ``0 .. n``."""
    if index is None:
        index = getattr(report, "index", None)
    if index is None:
        return np.arange(n)
    index = np.asarray(index)
    if len(index) != n:
        raise ValueError(f"index has length {len(index)}, expected {n}")
    return index


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of the finite values in each window of ``window`` bars."""
    if window < 1:
        raise ValueError("window must be 1 or more")
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    if len(values) < window:
        return out
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    csum = np.cumsum(clean)
    ccount = np.cumsum(finite)
    total = csum[window - 1 :].copy()
    count = ccount[window - 1 :].astype(np.float64, copy=True)
    total[1:] -= csum[: len(values) - window]
    count[1:] -= ccount[: len(values) - window]
    good = count > 0
    out[window - 1 :][good] = total[good] / count[good]
    return out


def _group_color(group: int) -> str:
    return _QUANTILES[group % len(_QUANTILES)]


def _series(pf: Any, method: str, account: int) -> np.ndarray:
    f = getattr(pf, method)
    return f(group=account) if pf.cash_sharing else f(account)


def value(pf: Any, account: int = 0, *, index: Any = None, title: str | None = None) -> Any:
    """The account's value, and its drawdown from the running peak below
    it."""
    go = _go()
    from plotly.subplots import make_subplots

    v = _series(pf, "value", account)
    x = _x(len(v), index)
    peak = np.fmax.accumulate(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = v / peak - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=x, y=v, name="value", line=dict(color=_LINE)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=dd, name="drawdown", fill="tozeroy", line=dict(color=_DOWN)),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(title=title or f"account {account}", hovermode="x unified")
    return fig


def orders(pf: Any, col: int = 0, *, index: Any = None, title: str | None = None) -> Any:
    """Column ``col``'s close with its fills: buys up, sells down."""
    go = _go()
    m = pf.column_map
    close = pf.close[:, col if m is None else m[col]]
    x = _x(len(close), index)
    o = pf.orders()
    mine = np.asarray(o["col"]) == col
    fig = go.Figure(go.Scatter(x=x, y=close, name="close", line=dict(color=_MUTED)))
    for side, name, color, symbol in ((0, "buy", _UP, "triangle-up"), (1, "sell", _DOWN, "triangle-down")):
        k = mine & (np.asarray(o["side"]) == side)
        fig.add_trace(go.Scatter(
            x=x[np.asarray(o["idx"])[k]], y=np.asarray(o["price"])[k], mode="markers", name=name,
            marker=dict(color=color, symbol=symbol, size=9),
            customdata=np.asarray(o["size"])[k], hovertemplate="%{y} x %{customdata}<extra>" + name + "</extra>",
        ))
    fig.update_layout(title=title or f"column {col}", hovermode="closest")
    return fig


def drawdowns(pf: Any, account: int = 0, *, index: Any = None, top: int = 5, title: str | None = None) -> Any:
    """The account's value with its ``top`` deepest drawdowns shaded."""
    go = _go()
    v = _series(pf, "value", account)
    x = _x(len(v), index)
    d = pf.drawdowns()
    mine = np.flatnonzero(np.asarray(d["col"]) == account)
    worst = mine[np.argsort(np.asarray(d["drawdown"])[mine])][:top]
    fig = go.Figure(go.Scatter(x=x, y=v, name="value", line=dict(color=_LINE)))
    for i in worst:
        a = int(d["peak_idx"][i])
        b = int(min(d["end_idx"][i], len(v) - 1))
        fig.add_vrect(x0=x[a], x1=x[b], fillcolor=_DOWN, opacity=0.12, line_width=0)
    fig.update_layout(title=title or f"account {account}: deepest drawdowns")
    return fig


def rolling(pf: Any, metric: str, window: int, *, index: Any = None, title: str | None = None, **kwargs: Any) -> Any:
    """``pf.rolling(metric, window)`` for every account."""
    go = _go()
    r = pf.rolling(metric, window, **kwargs)
    x = _x(r.shape[0], index)
    fig = go.Figure([go.Scatter(x=x, y=r[:, a], name=f"account {a}") for a in range(r.shape[1])])
    fig.update_layout(title=title or f"rolling {metric} ({window} bars)")
    return fig


def factor_ic(
    report: Any,
    factor: str,
    h: int = 1,
    method: str = "rank",
    *,
    index: Any = None,
    window: int = 20,
) -> Any:
    """IC of ``factor`` on each bar, and its trailing mean.

    Bars are ``report.ic(factor, h, method)``. ``window`` is the trailing
    mean drawn over them. ``index`` puts times on the x axis; omitted, the
    report's index is used.
    """
    go = _go()
    ic = np.asarray(report.ic(factor, h, method), dtype=np.float64)
    x = _axis(len(ic), index, report)
    roll = _rolling_mean(ic, int(window))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=ic, name="IC", marker=dict(color=_LINE)))
    fig.add_trace(go.Scatter(x=x, y=roll, name=f"{int(window)}-bar mean", line=dict(color=_UP)))
    fig.update_layout(title=f"{factor} IC (h={h}, {method})", hovermode="x unified")
    return fig


def factor_decay(report: Any, factor: str, method: str = "rank") -> Any:
    """Mean IC of ``factor`` at each horizon."""
    go = _go()
    decay = report.decay(factor, method)
    if hasattr(decay, "index"):
        horizons = list(decay.index)
        values = np.asarray(decay, dtype=np.float64)
    else:
        horizons = list(decay)
        values = np.asarray([decay[horizon] for horizon in horizons], dtype=np.float64)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=horizons, y=values, name="mean IC", marker=dict(color=_LINE)))
    fig.add_trace(go.Scatter(
        x=horizons, y=values, name="mean IC", mode="lines+markers",
        line=dict(color=_UP), showlegend=False,
    ))
    fig.update_layout(title=f"{factor} IC decay ({method})", xaxis_title="horizon", yaxis_title="mean IC")
    return fig


def factor_quantiles(
    report: Any,
    factor: str,
    h: int = 1,
    q: int = 5,
    *,
    index: Any = None,
) -> Any:
    """Mean return of each quantile, and each group's cumulative return.

    The left panel is the average forward return of groups ``0 .. q-1``. The
    right panel compounds those returns. A missing bar contributes nothing
    to the compound and breaks the line.
    """
    go = _go()
    from plotly.subplots import make_subplots

    table = report.quantile_returns(factor, h=h, q=q)
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"quantile returns must be 2-D; got {arr.ndim} dimensions")
    x = _axis(arr.shape[0], index, report)
    means = np.empty(arr.shape[1], dtype=np.float64)
    for g in range(arr.shape[1]):
        finite = arr[:, g]
        finite = finite[np.isfinite(finite)]
        means[g] = float(finite.mean()) if finite.size else np.nan
    valid = np.isfinite(arr)
    growth = np.where(valid, 1.0 + arr, 1.0)
    cumulative = np.cumprod(growth, axis=0) - 1.0
    cumulative = np.where(valid, cumulative, np.nan)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean return", "Cumulative return"))
    fig.add_trace(go.Bar(
        x=[str(g) for g in range(arr.shape[1])],
        y=means,
        name="mean",
        marker=dict(color=[_group_color(g) for g in range(arr.shape[1])]),
        showlegend=False,
    ), row=1, col=1)
    for g in range(arr.shape[1]):
        fig.add_trace(go.Scatter(
            x=x, y=cumulative[:, g], name=str(g), mode="lines", line=dict(color=_group_color(g)),
        ), row=1, col=2)
    fig.update_layout(title=f"{factor} quantiles (h={h}, q={q})")
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=2)
    return fig


def factor_turnover(
    report: Any,
    factor: str,
    q: int = 5,
    lag: int = 1,
    *,
    index: Any = None,
) -> Any:
    """Turnover of each quantile group over time."""
    go = _go()
    table = report.turnover(factor, q=q, lag=lag)
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"turnover must be 2-D; got {arr.ndim} dimensions")
    x = _axis(arr.shape[0], index, report)
    fig = go.Figure([
        go.Scatter(x=x, y=arr[:, g], name=str(g), mode="lines", line=dict(color=_group_color(g)))
        for g in range(arr.shape[1])
    ])
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title=f"{factor} turnover (q={q}, lag={lag})", hovermode="x unified")
    return fig
