"""Backtesting a model's predictions, against baselines that need no model.

A model that predicts returns is worth something only if trading on its
predictions beats trading without it. ``backtest`` runs the predictions as
signals and, beside them, the same run with baselines: holding throughout,
and random signals as many and as regularly spaced as the model's
(``phasor.signals.random_like``). ``compare`` puts their metrics side by
side.

The predictions must be made without the bar's own future: a prediction at
bar ``t`` may use what is known at ``t``'s close, and is traded from there.
Labels from ``phasor.labels`` are targets to train on, never predictions.

Before any trading, ``power`` measures how much the predictions say about what
followed -- correlation, R², and the returns of the prediction's quantile
groups -- over time (one market timed) or across instruments at each bar (a
factor's IC); ``decay`` says how long that lasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["backtest", "compare", "decay", "power"]


def backtest(close: Any, predictions: Any, *, enter_above: float = 0.0, exit_below: float | None = None,
             baselines: tuple[str, ...] = ("hold", "random"), seed: int = 0, **kwargs: Any) -> dict:
    """Portfolios by name: ``"model"`` and each of ``baselines``.

    The model enters where a prediction is above ``enter_above`` and exits
    where one is below ``exit_below`` (``enter_above`` by default). NaN
    predictions are neither. ``"hold"`` enters on the first bar and never
    exits; ``"random"`` is ``random_like(entries, keep="gaps")`` with exits
    at the model's holding periods, shuffled the same way. Every other
    keyword goes to ``from_signals`` for all of them.
    """
    from . import from_signals
    from .signals import random_like

    pred = np.asarray(predictions, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if pred.shape != close.shape:
        raise ValueError(f"predictions must be shaped like close: {pred.shape} and {close.shape}")
    lower = enter_above if exit_below is None else exit_below
    with np.errstate(invalid="ignore"):
        entries = pred > enter_above
        exits = pred < lower
    out = {"model": from_signals(close, entries, exits, **kwargs)}
    for name in baselines:
        if name == "hold":
            hold = np.zeros_like(entries)
            hold[0] = True
            out[name] = from_signals(close, hold, np.zeros_like(entries), **kwargs)
        elif name == "random":
            out[name] = from_signals(
                close,
                random_like(entries, seed=seed, keep="gaps"),
                random_like(exits, seed=seed + 1, keep="gaps"),
                **kwargs,
            )
        else:
            raise ValueError(f"baselines are 'hold' and 'random'; got {name!r}")
    return out


def compare(portfolios: dict, metrics: tuple[str, ...] = ("total_return", "sharpe_ratio", "max_drawdown")) -> dict:
    """Each metric per portfolio: ``{metric: {name: value per account}}``."""
    return {m: {name: getattr(pf, m)() for name, pf in portfolios.items()} for m in metrics}


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks from 0, ties sharing the mean of their positions."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], values.size]
    for a, b in zip(starts, ends):
        if b - a > 1:
            ranks[order[a:b]] = (a + b - 1) / 2.0
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _groups(pred: np.ndarray, quantiles: int) -> np.ndarray:
    """Quantile group of each prediction, from 0: by average rank, so tied
    predictions (a position of exactly 1, say) always share a group."""
    return np.minimum((_ranks(pred) / pred.size * quantiles).astype(np.int64), quantiles - 1)


def _one_sample(pred: np.ndarray, fwd: np.ndarray, quantiles: int) -> dict:
    corr = _corr(pred, fwd)
    group = _groups(pred, quantiles)
    means = {}
    counts = {}
    for g in range(quantiles):
        k = group == g
        if k.any():
            means[g + 1] = float(fwd[k].mean())
            counts[g + 1] = int(k.sum())
    return {
        "correlation": corr,
        "rank_correlation": _corr(_ranks(pred), _ranks(fwd)),
        "r_squared": corr * corr,
        "quantile_returns": means,
        "quantile_counts": counts,
    }


def _monotonicity(means: dict) -> float:
    if len(means) < 2:
        return float("nan")
    keys = np.array(sorted(means), dtype=np.float64)
    return _corr(_ranks(keys), _ranks(np.array([means[k] for k in sorted(means)])))


def power(predictions: Any, forward_returns: Any, *, quantiles: int = 5, axis: str = "time") -> dict:
    """How much the predictions say about the returns that followed them.

    ``predictions[t]`` was known at ``t``; ``forward_returns[t]`` is what
    followed it (``phasor.labels.fixed_horizon_returns``, say). Both have one
    shape, ``(bars,)`` or ``(bars, instruments)``; a pair with a ``NaN`` on
    either side is left out.

    ``axis="time"`` pools every pair, as when one market is timed:
    ``correlation``, ``rank_correlation``, ``r_squared`` (the share of the
    returns' variance the predictions explain linearly; 0.05 out of sample is
    exceptional for a return forecast), the mean return of each of
    ``quantiles`` groups of predictions (``quantile_returns``, from the most
    bearish), their ``quantile_counts``, and ``monotonicity``: the rank
    correlation between the group and its return, 1 when each group did better
    than the one below. Tied predictions share a group, so a position of -1, 0
    or 1 gives three groups whatever ``quantiles`` is.

    ``axis="cross_section"`` compares instruments with each other at each bar
    and averages over the bars -- a factor's information coefficient: the same
    keys as the mean over bars, plus ``correlation_std`` and
    ``information_ratio`` (the mean correlation over its standard deviation)
    and the number of ``periods``. A bar with fewer than two instruments is
    skipped.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    fwd = np.asarray(forward_returns, dtype=np.float64)
    if pred.shape != fwd.shape:
        raise ValueError(f"predictions and forward_returns differ in shape: {pred.shape} and {fwd.shape}")
    if quantiles < 1:
        raise ValueError(f"quantiles must be at least 1; got {quantiles}")
    keep = np.isfinite(pred) & np.isfinite(fwd)
    if axis == "time":
        out = _one_sample(pred[keep], fwd[keep], quantiles)
        out["monotonicity"] = _monotonicity(out["quantile_returns"])
        out["observations"] = int(keep.sum())
        return out
    if axis != "cross_section":
        raise ValueError(f"axis is 'time' or 'cross_section'; got {axis!r}")
    if pred.ndim != 2:
        raise ValueError("axis='cross_section' needs (bars, instruments) arrays")
    rows = [_one_sample(pred[t, keep[t]], fwd[t, keep[t]], quantiles) for t in range(len(pred)) if keep[t].sum() >= 2]
    corr = np.array([r["correlation"] for r in rows], dtype=np.float64)
    finite = corr[np.isfinite(corr)]
    mean_of = lambda key: float(np.nanmean([r[key] for r in rows])) if rows else float("nan")  # noqa: E731
    groups = sorted({g for r in rows for g in r["quantile_returns"]})
    means = {g: float(np.mean([r["quantile_returns"][g] for r in rows if g in r["quantile_returns"]])) for g in groups}
    std = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    return {
        "correlation": float(finite.mean()) if finite.size else float("nan"),
        "rank_correlation": mean_of("rank_correlation"),
        "r_squared": mean_of("r_squared"),
        "quantile_returns": means,
        "quantile_counts": {g: int(sum(r["quantile_counts"].get(g, 0) for r in rows)) for g in groups},
        "monotonicity": _monotonicity(means),
        "correlation_std": std,
        "information_ratio": float(finite.mean() / std) if finite.size > 1 and std > 0 else float("nan"),
        "periods": len(rows),
        "observations": int(keep.sum()),
    }


def decay(predictions: Any, close: Any, horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> dict[int, float]:
    """The mean return in the direction of the prediction, ``h`` bars on:
    ``{h: mean of sign(prediction[t]) * (close[t + h] / close[t] - 1)}`` over
    the non-zero predictions, before costs. How fast it rises and then fades
    says how long the information lasts: a horizon past the peak is time the
    position is held for nothing."""
    pred = np.asarray(predictions, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if pred.shape != close.shape:
        raise ValueError(f"predictions must be shaped like close: {pred.shape} and {close.shape}")
    side = np.sign(np.nan_to_num(pred))
    out = {}
    for h in horizons:
        if int(h) < 1:
            raise ValueError(f"horizons must be at least 1; got {h}")
        h = int(h)
        forward = np.full(close.shape, np.nan)
        if h < len(close):
            with np.errstate(divide="ignore", invalid="ignore"):
                forward[:-h] = close[h:] / close[:-h] - 1
        values = (side * forward)[(side != 0) & np.isfinite(forward)]
        out[h] = float(values.mean()) if values.size else float("nan")
    return out
