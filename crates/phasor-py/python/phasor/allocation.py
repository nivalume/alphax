"""Portfolio weights on a rebalancing schedule, and their backtest.

A weight function sees the trailing window of returns, ``(lookback, n_assets)``,
and returns one weight per asset. :func:`weights` calls it on each
rebalancing bar and leaves every other bar ``NaN``; :func:`from_weights`
turns that matrix into target-percent orders on one shared-cash account::

    from phasor import allocation as al

    w = al.weights(close, al.min_variance, every=21, lookback=126)
    pf = al.from_weights(close, w, fees=0.0005)
    base = al.from_weights(close, al.weights(close, al.equal_weight, every=21))

    w = al.weights(close, al.from_pypfopt("max_sharpe"), freq="M", index=dates)

A weight function only ever sees rows before the bar it rebalances on (the
window ends at the previous bar), so the orders placed at a bar's close never
use that bar's return. Weights need not sum to one: the rest is cash, and a
negative weight is a short.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "equal_weight",
    "from_pypfopt",
    "from_riskfolio",
    "from_weights",
    "inverse_volatility",
    "marginal_contribution",
    "max_sharpe",
    "min_variance",
    "rebalance_mask",
    "risk_parity",
    "universal_weights",
    "weights",
]

WeightFn = Callable[[np.ndarray], Any]


def _close(close: Any) -> np.ndarray:
    a = np.asarray(close, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError("close must be 1-D or 2-D")
    return a


def rebalance_mask(n: int, *, every: int | None = None, freq: str | None = None,
                   index: Any = None, offset: int = 0) -> np.ndarray:
    """Which of ``n`` bars rebalance: every ``every`` bars from ``offset``,
    or the first bar of each calendar period (``freq`` = ``W``, ``M``,
    ``Q``, ``Y``; needs ``index``)."""
    if (every is None) == (freq is None):
        raise ValueError("give exactly one of every and freq")
    mask = np.zeros(n, dtype=bool)
    if every is not None:
        if every < 1:
            raise ValueError("every must be at least 1")
        mask[offset::every] = True
        return mask
    if index is None:
        raise ValueError("freq needs index, the bars' times")
    from .validation import _period_ids, _times

    ids = _period_ids(_times(index), freq)
    if len(ids) != n:
        raise ValueError(f"index has {len(ids)} times for {n} bars")
    mask[np.r_[True, ids[1:] != ids[:-1]]] = True
    return mask


def weights(close: Any, fn: WeightFn, *, every: int | None = None, freq: str | None = None,
            index: Any = None, lookback: int = 60, min_periods: int | None = None) -> np.ndarray:
    """Target weights, ``(n_bars, n_assets)``: ``fn(window)`` on each
    rebalancing bar with at least ``min_periods`` (default ``lookback``) past
    returns, ``NaN`` elsewhere. ``window`` holds the simple returns of the
    ``lookback`` bars before this one, ``NaN`` where a price is missing."""
    c = _close(close)
    n, k = c.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.vstack([np.full((1, k), np.nan), c[1:] / c[:-1] - 1])
    mask = rebalance_mask(n, every=every, freq=freq, index=index)
    need = lookback if min_periods is None else min_periods
    out = np.full((n, k), np.nan)
    for t in np.flatnonzero(mask):
        # Returns 1..t-1 are known at bar t's open; bar t's own is not used.
        window = ret[max(1, t - lookback):t]
        if len(window) < need:
            continue
        w = np.asarray(fn(window), dtype=float).ravel()
        if w.shape != (k,):
            raise ValueError(f"the weight function returned {w.shape[0]} weights for {k} assets")
        out[t] = w
    return out


def from_weights(close: Any, weights: Any, **kwargs: Any) -> Any:
    """Backtests target weights: ``phasor.from_orders`` with
    ``size_type="targetpercent"`` on one shared-cash account, sells before
    buys (``call_seq="auto"``). ``NaN`` rows place no orders. Other keyword
    arguments go to ``from_orders``."""
    from . import from_orders

    c = _close(close)
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    if w.shape != c.shape:
        raise ValueError(f"weights {w.shape} and close {c.shape} differ in shape")
    kwargs.setdefault("call_seq", "auto")
    return from_orders(c, np.asfortranarray(w), size_type="targetpercent", cash_sharing=True,
                       group_lens=[c.shape[1]], **kwargs)


# ------------------------------------------------------------ weight functions


def _present(window: np.ndarray) -> np.ndarray:
    """Rows where every asset has a return."""
    return window[~np.isnan(window).any(axis=1)]


def _cov(window: np.ndarray) -> np.ndarray:
    r = _present(window)
    if len(r) < 2:
        raise ValueError("fewer than two complete rows of returns in the window")
    s = np.cov(r, rowvar=False, ddof=1)
    s = np.atleast_2d(s)
    # A small ridge keeps a singular estimate invertible.
    return s + np.eye(len(s)) * 1e-10 * max(np.trace(s) / len(s), 1e-300)


def equal_weight(window: np.ndarray) -> np.ndarray:
    """``1 / n`` each."""
    k = window.shape[1]
    return np.full(k, 1.0 / k)


def inverse_volatility(window: np.ndarray) -> np.ndarray:
    """Proportional to ``1 / std``, summing to one."""
    sd = np.nanstd(window, axis=0, ddof=1)
    inv = np.where(sd > 0, 1.0 / sd, 0.0)
    return inv / inv.sum() if inv.sum() > 0 else equal_weight(window)


def _solve_long_only(s: np.ndarray, b: np.ndarray, long_only: bool) -> np.ndarray:
    """``w ∝ S⁻¹ b`` on the active assets, dropping those that come out
    negative until none do (``long_only``); normalized to sum to one."""
    k = len(b)
    active = np.ones(k, dtype=bool)
    while True:
        w = np.zeros(k)
        w[active] = np.linalg.solve(s[np.ix_(active, active)], b[active])
        total = w.sum()
        if total <= 0:
            return np.zeros(k)
        w /= total
        if not long_only or (w >= -1e-15).all():
            return np.where(np.abs(w) < 1e-15, 0.0, w)
        active &= w > 0
        if not active.any():
            return np.zeros(k)


def min_variance(window: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Minimum variance under the sample covariance, fully invested."""
    s = _cov(window)
    return _solve_long_only(s, np.ones(len(s)), long_only)


def max_sharpe(window: np.ndarray, risk_free: float = 0.0, long_only: bool = True) -> np.ndarray:
    """The tangency portfolio, ``w ∝ S⁻¹ (μ - rf)``, fully invested; all cash
    (zeros) when no asset beats ``risk_free``."""
    r = _present(window)
    mu = r.mean(axis=0) - risk_free
    if (mu <= 0).all():
        return np.zeros(window.shape[1])
    return _solve_long_only(_cov(window), mu, long_only)


def risk_parity(window: np.ndarray, iterations: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Equal risk contributions, ``w_i (S w)_i`` the same for every asset,
    long only, summing to one.

    Solved by cyclical coordinate descent on ``½ wᵀSw - Σ log(w_i) / k``
    (Spinu, 2013): each pass sets every ``w_i`` to the positive root of its
    own quadratic, so a weight is never negative and no square root of a
    negative risk contribution is ever taken, whatever the covariances'
    signs. The minimizer has ``w_i (S w)_i`` equal for every asset; it is
    then scaled to sum to one.
    """
    s = _cov(window)
    k = len(s)
    b = 1.0 / k
    w = np.full(k, 1.0 / k)
    diag = np.diag(s)
    for _ in range(iterations):
        before = w / w.sum()
        for i in range(k):
            alpha = s[i] @ w - diag[i] * w[i]
            w[i] = (-alpha + np.sqrt(alpha * alpha + 4.0 * diag[i] * b)) / (2.0 * diag[i])
        after = w / w.sum()
        if np.abs(after - before).max() < tol:
            return after
    return w / w.sum()


# ------------------------------------------------------------ library wrappers


def _frame(window: np.ndarray) -> Any:
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("the optimizer wrappers need pandas") from e
    return pd.DataFrame(_present(window), columns=[f"a{i}" for i in range(window.shape[1])])


def from_pypfopt(method: str = "max_sharpe", *, frequency: int = 252, **method_kwargs: Any) -> WeightFn:
    """A weight function from PyPortfolioOpt: historical mean and sample
    covariance, then ``EfficientFrontier.<method>(**method_kwargs)``
    (``max_sharpe``, ``min_volatility``, ``efficient_risk``, ...)."""

    def fn(window: np.ndarray) -> np.ndarray:
        from pypfopt import EfficientFrontier, expected_returns, risk_models

        df = _frame(window)
        mu = expected_returns.mean_historical_return(df, returns_data=True, frequency=frequency)
        cov = risk_models.sample_cov(df, returns_data=True, frequency=frequency)
        ef = EfficientFrontier(mu, cov)
        getattr(ef, method)(**method_kwargs)
        w = ef.clean_weights()
        return np.array([w[c] for c in df.columns])

    return fn


def from_riskfolio(*, model: str = "Classic", rm: str = "MV", obj: str = "Sharpe",
                   rf: float = 0.0, **kwargs: Any) -> WeightFn:
    """A weight function from Riskfolio-Lib: historical estimates, then
    ``Portfolio.optimization(model, rm, obj, rf, ...)``."""

    def fn(window: np.ndarray) -> np.ndarray:
        import riskfolio as rp

        df = _frame(window)
        port = rp.Portfolio(returns=df)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, hist=True, **kwargs)
        if w is None:
            return np.full(window.shape[1], np.nan)
        return w.reindex(df.columns).to_numpy().ravel()

    return fn


def universal_weights(close: Any, algo: str = "OLMAR", **kwargs: Any) -> np.ndarray:
    """Weights from an online algorithm of the Universal Portfolios library
    (``universal.algos.<algo>``), one row per bar. These algorithms decide a
    bar's weights from prices up to the bar before, as the library runs
    them; pass the result to :func:`from_weights`."""
    import pandas as pd
    from universal import algos

    c = _close(close)
    result = getattr(algos, algo)(**kwargs).run(pd.DataFrame(c))
    return np.asarray(result.weights, dtype=float)


def _mask(mask: Any, n: int, name: str) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape != (n,):
        raise ValueError(f"{name} must be a boolean mask of {n} rows; got shape {m.shape}")
    return m


def marginal_contribution(
    book: Any,
    candidate: Any,
    *,
    fit: Any = None,
    rows: Any = None,
    ann_factor: float = 365.0,
) -> dict[str, Any]:
    """What adding a strategy does to a book of strategies already run.

    ``book`` is the strategies' per-bar returns, ``(bars, k)`` (``k`` may be
    0, or ``None`` for an empty book); ``candidate`` is ``(bars,)``. Each
    strategy is scaled to unit volatility over the ``fit`` rows (a boolean
    mask; every row by default) and the book is their equal-weight average,
    so a strategy counts by its risk, not its leverage. Over the ``rows``
    mask (every row by default) it returns:

    - ``correlations``: the candidate's correlation with each strategy, and
      ``max_abs_correlation``, the largest in size;
    - ``correlation_to_book``: with the book as a whole;
    - ``candidate_sharpe``, ``book_sharpe`` and ``combined_sharpe`` (the book
      with the candidate as one more equal member), annualized, and ``gain``
      = combined minus book. With an empty book the gain is the candidate's
      own Sharpe.

    A strategy that is good alone but adds nothing to the book -- high
    correlation, no gain -- is a copy of what is already run. Missing returns
    count as zero (the strategy was flat).
    """
    from . import returns as _returns

    cand = np.nan_to_num(np.asarray(candidate, dtype=np.float64).ravel())
    n = cand.size
    if book is None:
        b = np.zeros((n, 0))
    else:
        b = np.nan_to_num(np.asarray(book, dtype=np.float64))
        b = b[:, None] if b.ndim == 1 else b
    if b.shape[0] != n:
        raise ValueError(f"book and candidate differ in length: {b.shape[0]} and {n}")
    fit_rows = _mask(fit, n, "fit")
    eval_rows = _mask(rows, n, "rows")

    def scaled(x: np.ndarray) -> np.ndarray:
        sd = x[fit_rows].std(axis=0, ddof=1) if fit_rows.sum() > 1 else np.full(x.shape[1:], np.nan)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
        return np.nan_to_num(x / sd)

    def sharpe(x: np.ndarray) -> float:
        return float(_returns.metric(x[eval_rows], "sharpe_ratio", ann_factor=ann_factor))

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        x, y = x[eval_rows], y[eval_rows]
        if x.size < 2 or x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    c = scaled(cand[:, None])[:, 0]
    members = scaled(b)
    k = members.shape[1]
    correlations = [corr(c, members[:, j]) for j in range(k)]
    finite = [abs(v) for v in correlations if np.isfinite(v)]
    candidate_sharpe = sharpe(cand)
    if k == 0:
        return {
            "correlations": [], "max_abs_correlation": float("nan"), "correlation_to_book": float("nan"),
            "candidate_sharpe": candidate_sharpe, "book_sharpe": float("nan"),
            "combined_sharpe": candidate_sharpe, "gain": candidate_sharpe,
        }
    book_returns = members.mean(axis=1)
    combined = (members.sum(axis=1) + c) / (k + 1)
    book_sharpe, combined_sharpe = sharpe(book_returns), sharpe(combined)
    return {
        "correlations": correlations,
        "max_abs_correlation": max(finite) if finite else float("nan"),
        "correlation_to_book": corr(c, book_returns),
        "candidate_sharpe": candidate_sharpe,
        "book_sharpe": book_sharpe,
        "combined_sharpe": combined_sharpe,
        "gain": combined_sharpe - book_sharpe,
    }
