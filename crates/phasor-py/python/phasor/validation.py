"""Time-series cross-validation: which rows a parameter choice may see.

A backtest that picks its parameters on the same bars it reports is measuring
its own fit, not its edge. The splitters here return row indices, not data, so
they compose with anything that takes an index: slicing a price matrix before
``phasor.from_signals``, a pandas ``iloc``, or a model's ``fit``.

    import numpy as np
    from phasor import validation

    for split in validation.rolling(n=1000, train=500, test=100):
        close[split.train], close[split.test]

Every splitter keeps time order: test rows always come from a contiguous block,
and for ``rolling`` and ``expanding`` every train row precedes every test row.
``purged_kfold`` is the one exception, because k-fold uses the future to train
on the past; it pays for that by dropping the rows where leakage concentrates.

Splitters are explicit about their sizes rather than accepting a fraction or a
count of splits, so the same call on a longer series gives more splits of the
same shape, not a different shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Split",
    "calendar",
    "expanding",
    "from_masks",
    "from_ranges",
    "permutation_test",
    "position_delay_test",
    "purged_kfold",
    "rolling",
    "shuffled_prices",
    "shuffled_test",
    "signal_delay_test",
    "walk_forward",
]


@dataclass(frozen=True, eq=False)
class Split:
    """One train/test pair of integer row indices.

    Attributes
    ----------
    train : np.ndarray
        Sorted ``int64`` row indices to fit or select parameters on.
    test : np.ndarray
        Sorted ``int64`` row indices to evaluate the chosen parameters on.
        Never overlaps ``train``.
    """

    train: np.ndarray
    test: np.ndarray

    def __repr__(self) -> str:
        return f"Split(train={_span(self.train)}, test={_span(self.test)})"


def _span(index: np.ndarray) -> str:
    """A compact description: a range when contiguous, a count otherwise."""
    if index.size == 0:
        return "[]"
    if index[-1] - index[0] + 1 == index.size:
        return f"[{index[0]}:{index[-1] + 1}]"
    return f"<{index.size} rows in [{index[0]}:{index[-1] + 1}]>"


def _int(name: str, value: Any, minimum: int) -> int:
    """``value`` as an int no smaller than ``minimum``, or a ``ValueError``.

    ``bool`` is refused even though it is an ``int`` subclass: ``test=True``
    is a mistake, not a window of one row.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _range(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.int64)


def rolling(n: int, train: int, test: int, step: int | None = None) -> list[Split]:
    """Fixed-length train windows, each followed by a test window.

    Split ``i`` trains on rows ``[s, s + train)`` and tests on
    ``[s + train, s + train + test)`` where ``s = i * step``. A fixed train
    length means every parameter choice is made on the same amount of history,
    so a drift in the chosen parameters reflects the market, not a growing
    sample.

    Parameters
    ----------
    n : int
        Number of rows in the series.
    train : int
        Rows in each train window (>= 1).
    test : int
        Rows in each test window (>= 1).
    step : int, optional
        Rows to advance between splits. Defaults to ``test``, which makes the
        test windows tile the series without overlap, so their results can be
        concatenated into one out-of-sample record.

    Returns
    -------
    list of Split
        In time order. Trailing rows that cannot fill a whole test window are
        left out rather than tested on a short window.

    Raises
    ------
    ValueError
        If an argument is not a positive integer, or ``n < train + test``.
    """
    n = _int("n", n, 0)
    train = _int("train", train, 1)
    test = _int("test", test, 1)
    step = test if step is None else _int("step", step, 1)
    if train + test > n:
        raise ValueError(
            f"train + test ({train} + {test}) exceeds n ({n}); no split fits"
        )
    return [
        Split(_range(s, s + train), _range(s + train, s + train + test))
        for s in range(0, n - train - test + 1, step)
    ]


def expanding(n: int, min_train: int, test: int, step: int | None = None) -> list[Split]:
    """Train windows anchored at row 0 that grow by ``step`` each split.

    Split ``i`` trains on rows ``[0, e)`` and tests on ``[e, e + test)`` where
    ``e = min_train + i * step``. Use this when older history stays relevant
    and more of it should only sharpen the estimate; use ``rolling`` when old
    regimes should age out.

    Parameters
    ----------
    n : int
        Number of rows in the series.
    min_train : int
        Rows in the first (shortest) train window (>= 1).
    test : int
        Rows in each test window (>= 1).
    step : int, optional
        Rows the train window grows by between splits. Defaults to ``test``,
        so test windows tile the series without overlap.

    Returns
    -------
    list of Split
        In time order; every ``train`` starts at row 0.

    Raises
    ------
    ValueError
        If an argument is not a positive integer, or ``n < min_train + test``.
    """
    n = _int("n", n, 0)
    min_train = _int("min_train", min_train, 1)
    test = _int("test", test, 1)
    step = test if step is None else _int("step", step, 1)
    if min_train + test > n:
        raise ValueError(
            f"min_train + test ({min_train} + {test}) exceeds n ({n}); no split fits"
        )
    return [
        Split(_range(0, e), _range(e, e + test))
        for e in range(min_train, n - test + 1, step)
    ]


def purged_kfold(n: int, k: int, purge: int = 0, embargo: int = 0) -> list[Split]:
    """K contiguous test folds; train on the rest, minus the leaky edges.

    The series is cut into ``k`` contiguous folds (sizes differ by at most one
    row; the first ``n % k`` folds get the extra row). Each fold is a test set
    once, and trains on every other row except:

    * **purge** -- the ``purge`` rows immediately *before* the test fold. A
      label at row ``t`` usually looks forward (a return over the next ``h``
      bars, a triple-barrier outcome), so a train row just before the test
      fold has a label computed from test-period prices. Training on it fits
      the answer. Set ``purge`` to the label horizon.
    * **embargo** -- the ``embargo`` rows immediately *after* the test fold.
      Features usually look backward (moving averages, rolling volatility), so
      a train row just after the test fold has features built from test-period
      prices, and serial correlation makes its label close to the last test
      labels. Set ``embargo`` to the longest feature lookback, or a small
      fraction of ``n``.

    This follows López de Prado, *Advances in Financial Machine Learning*
    (2018), ch. 7, with the label horizon given as a fixed row count.

    Parameters
    ----------
    n : int
        Number of rows in the series.
    k : int
        Number of folds (2 <= k <= n).
    purge : int, default 0
        Rows dropped from train before each test fold.
    embargo : int, default 0
        Rows dropped from train after each test fold.

    Returns
    -------
    list of Split
        One per fold, in time order. ``train`` is generally not contiguous.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``purge`` and ``embargo`` leave
        some fold with no train rows.
    """
    n = _int("n", n, 0)
    k = _int("k", k, 2)
    purge = _int("purge", purge, 0)
    embargo = _int("embargo", embargo, 0)
    if k > n:
        raise ValueError(f"k ({k}) exceeds n ({n}); some fold would be empty")
    rows = _range(0, n)
    splits = []
    for fold in np.array_split(rows, k):
        start, stop = int(fold[0]), int(fold[-1]) + 1
        keep = (rows < start - purge) | (rows >= stop + embargo)
        if not keep.any():
            raise ValueError(
                f"purge ({purge}) and embargo ({embargo}) leave no train rows "
                f"for the test fold [{start}:{stop}]"
            )
        splits.append(Split(rows[keep], fold.copy()))
    return splits


def _argmax(scores: Mapping[Hashable, float]) -> Hashable:
    """The key with the highest finite score; NaN and infinite scores never
    win, ties go first.

    Infinity is skipped for the same reason as NaN: it is not a measurement.
    A Sharpe ratio is ``+inf`` for a window whose returns never moved — a
    parameter that never traded there — and letting it win would select the
    one combination that did nothing.
    """
    best_key, best = None, -math.inf
    found = False
    for key, score in scores.items():
        score = float(score)
        if not math.isfinite(score):
            continue
        if not found or score > best:
            best_key, best, found = key, score, True
    if not found:
        raise ValueError("every train score is NaN or infinite; nothing to select")
    return best_key


def walk_forward(
    fn: Callable[[np.ndarray], Mapping[Hashable, float]],
    n: int,
    splits: Iterable[Split],
    *,
    select: Callable[[Mapping[Hashable, float]], Hashable] | None = None,
) -> list[dict[str, Any]]:
    """Select parameters on each train window, report them on its test window.

    ``fn`` evaluates a whole parameter grid on a set of rows and returns
    ``{param_key: score}``. For each split, ``fn(split.train)`` scores the
    grid, ``select`` picks one key, then ``fn(split.test)`` scores the grid
    out of sample and only the chosen key's score is kept. The other test
    scores are discarded so they cannot influence the choice: reporting the
    best *test* score would be the in-sample fit this exists to avoid.

    Nothing here runs a backtest, so ``fn`` can wrap ``phasor.from_signals``
    over a parameter grid (one column per parameter set) or anything else::

        def fn(rows):
            pf = phasor.from_signals(close[rows], entries[rows], exits[rows])
            return dict(zip(params, pf.total_return()))

    Parameters
    ----------
    fn : callable
        ``fn(rows) -> Mapping[key, float]`` where ``rows`` is an ``int64``
        index array. Must return the chosen key on the test rows too.
    n : int
        Number of rows in the series; every split index must lie in
        ``[0, n)``. Catches splits built for a different series.
    splits : iterable of Split
        From ``rolling``, ``expanding``, ``purged_kfold``, or hand-built.
    select : callable, optional
        ``select(train_scores) -> key``. Defaults to the key with the highest
        score (NaN and ±inf never win; ties go to the first key in ``fn``'s order).
        Pass e.g. ``lambda s: min(s, key=s.get)`` to minimize a loss.

    Returns
    -------
    list of dict
        One per split, in order, with keys ``split`` (the position of the
        split in ``splits``), ``chosen`` (the selected key), ``train_score``
        and ``test_score`` (that key's score on train and on test).

    Raises
    ------
    ValueError
        If ``splits`` is empty, a split's indices fall outside ``[0, n)`` or
        overlap, ``fn`` returns no scores, or ``select`` picks a key missing
        from the train or test scores.
    """
    if not callable(fn):
        raise ValueError(f"fn must be callable; got {fn!r}")
    n = _int("n", n, 1)
    select = _argmax if select is None else select
    splits = list(splits)
    if not splits:
        raise ValueError("splits is empty; nothing to walk forward over")

    results = []
    for i, split in enumerate(splits):
        if not isinstance(split, Split):
            raise ValueError(f"splits[{i}] must be a Split; got {type(split).__name__}")
        for name in ("train", "test"):
            rows = getattr(split, name)
            if rows.size == 0:
                raise ValueError(f"splits[{i}].{name} is empty")
            if rows.min() < 0 or rows.max() >= n:
                raise ValueError(f"splits[{i}].{name} has indices outside [0, {n})")
        if np.intersect1d(split.train, split.test).size:
            raise ValueError(f"splits[{i}] has rows in both train and test")

        train_scores = fn(split.train)
        if not train_scores:
            raise ValueError(f"fn returned no scores for splits[{i}].train")
        chosen = select(train_scores)
        if chosen not in train_scores:
            raise ValueError(f"select chose {chosen!r}, which is not a train score key")
        test_scores = fn(split.test)
        if chosen not in test_scores:
            raise ValueError(
                f"fn did not score the chosen key {chosen!r} on splits[{i}].test"
            )
        results.append(
            {
                "split": i,
                "chosen": chosen,
                "train_score": train_scores[chosen],
                "test_score": test_scores[chosen],
            }
        )
    return results


# ------------------------------------------------------- splits from dates


def _times(index: Any) -> np.ndarray:
    index = np.asarray(index)
    if index.ndim != 1 or index.size == 0:
        raise ValueError("index must be a non-empty 1-D array of times")
    if index.dtype.kind != "M":
        index = index.astype("datetime64[ns]")
    if (np.diff(index.astype(np.int64)) < 0).any():
        raise ValueError("index must be sorted")
    return index


def _rows(index: np.ndarray, start: Any, stop: Any) -> np.ndarray:
    """Rows with ``start <= time < stop``."""
    lo = np.searchsorted(index, np.datetime64(start, "ns").astype(index.dtype), "left")
    hi = np.searchsorted(index, np.datetime64(stop, "ns").astype(index.dtype), "left")
    return _range(int(lo), int(hi))


def from_ranges(index: Any, ranges: Iterable[tuple[Any, Any, Any, Any]]) -> list[Split]:
    """Splits from date ranges: each ``(train_start, train_stop, test_start,
    test_stop)``, half-open, in any type ``numpy.datetime64`` accepts
    (``"2024-01-01"``, a ``datetime``). ``index`` is the bars' times.

    Raises
    ------
    ValueError
        A range with no bars, or train and test sharing a bar.
    """
    index = _times(index)
    out = []
    for i, (a, b, c, d) in enumerate(ranges):
        split = Split(_rows(index, a, b), _rows(index, c, d))
        if split.train.size == 0 or split.test.size == 0:
            raise ValueError(f"ranges[{i}] covers no bars on one side")
        if np.intersect1d(split.train, split.test).size:
            raise ValueError(f"ranges[{i}] puts the same bars in train and test")
        out.append(split)
    return out


def from_masks(train: Any, test: Any) -> list[Split]:
    """Splits from boolean masks, one row per split (or one 1-D mask each)."""
    train = np.atleast_2d(np.asarray(train, dtype=bool))
    test = np.atleast_2d(np.asarray(test, dtype=bool))
    if train.shape != test.shape:
        raise ValueError(f"train and test masks differ in shape: {train.shape} and {test.shape}")
    if (train & test).any():
        raise ValueError("a row is in both train and test")
    return [Split(np.flatnonzero(a).astype(np.int64), np.flatnonzero(b).astype(np.int64))
            for a, b in zip(train, test)]


_PERIODS = {"W": "W", "M": "M", "ME": "M", "Q": "Q", "QE": "Q", "Y": "Y", "YE": "Y", "A": "Y"}


def _period_ids(index: np.ndarray, freq: str) -> np.ndarray:
    unit = _PERIODS.get(freq.upper())
    if unit is None:
        raise ValueError(f"freq must be one of W, M, Q, Y; got {freq!r}")
    days = index.astype("datetime64[D]")
    if unit == "W":
        # Weeks starting Monday: 1970-01-01 was a Thursday.
        return (days.astype(np.int64) + 3) // 7
    months = days.astype("datetime64[M]").astype(np.int64)
    return {"M": months, "Q": months // 3, "Y": months // 12}[unit]


def calendar(index: Any, freq: str, train: int, test: int = 1, step: int | None = None,
             expanding: bool = False) -> list[Split]:
    """Splits on calendar periods: train on ``train`` whole periods (weeks,
    months, quarters or years, as ``freq``), test on the next ``test``,
    advance by ``step`` periods (default ``test``). With ``expanding`` the
    train window keeps its first period. Periods are those the bars fall in,
    so a month without bars is not a period."""
    index = _times(index)
    train = _int("train", train, 1)
    test = _int("test", test, 1)
    step = test if step is None else _int("step", step, 1)
    ids = _period_ids(index, freq)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    bounds = np.r_[starts, len(index)]
    n_periods = len(starts)
    out = []
    for first in range(0, n_periods - train - test + 1, step):
        lo = 0 if expanding else first
        mid, hi = first + train, first + train + test
        out.append(Split(_range(int(bounds[lo]), int(bounds[mid])), _range(int(bounds[mid]), int(bounds[hi]))))
    if not out:
        raise ValueError(f"{n_periods} periods cannot hold {train} train and {test} test periods")
    return out


# ------------------------------------------------------- significance tests


def permutation_test(
    positions: Any,
    returns: Any,
    *,
    statistic: str | Callable[[np.ndarray], float] = "total",
    n: int = 1000,
    block: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Is the strategy's edge more than chance alignment?

    ``positions[t]`` is the exposure held over bar ``t``'s return
    ``returns[t]`` (both 1-D, same length; ``NaN`` returns count as zero).
    The statistic of ``positions * returns`` is compared with its value when
    the returns are shuffled in blocks of ``block`` bars (``block > 1``
    keeps short-range autocorrelation), which keeps the positions' timing and
    the returns' distribution but breaks their alignment.

    Both may instead be ``(bars, assets)``: the strategy's return at ``t`` is
    then the sum over assets of ``positions * returns``, and every asset's
    returns are shuffled by the same permutation of bars, so their
    co-movement survives and only the timing is tested.

    ``statistic`` is ``"total"`` (sum), ``"mean"``, ``"sharpe"`` (mean / std)
    or a function of the strategy's returns.

    Returns ``observed``, the ``null`` distribution and ``p_value`` =
    ``(1 + #null >= observed) / (n + 1)``.
    """
    pos = np.asarray(positions, dtype=float)
    ret = np.nan_to_num(np.asarray(returns, dtype=float))
    if pos.ndim == 2 or ret.ndim == 2:
        if pos.shape != ret.shape or pos.ndim != 2:
            raise ValueError(f"positions and returns must have one shape; got {pos.shape} and {ret.shape}")

        def pnl(r: np.ndarray) -> np.ndarray:
            return (pos * r).sum(axis=1)
    else:
        pos, ret = pos.ravel(), ret.ravel()
        if pos.shape != ret.shape:
            raise ValueError(f"positions and returns differ in length: {pos.size} and {ret.size}")

        def pnl(r: np.ndarray) -> np.ndarray:
            return pos * r
    n = _int("n", n, 1)
    block = _int("block", block, 1)
    stats = {
        "total": np.sum,
        "mean": np.mean,
        "sharpe": lambda r: float(np.mean(r) / np.std(r, ddof=1)) if np.std(r, ddof=1) > 0 else 0.0,
    }
    f = stats.get(statistic) if isinstance(statistic, str) else statistic
    if f is None:
        raise ValueError(f"statistic must be one of {sorted(stats)} or a function")
    observed = float(f(pnl(ret)))
    rng = np.random.default_rng(seed)
    blocks = [ret[i:i + block] for i in range(0, len(ret), block)]
    null = np.empty(n)
    for k in range(n):
        shuffled = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        null[k] = f(pnl(shuffled))
    p = (1 + int((null >= observed).sum())) / (n + 1)
    return {"observed": observed, "null": null, "p_value": p}


def position_delay_test(
    positions: Any,
    returns: Any,
    delays: Iterable[int] = (0, 1, 2, 3),
    *,
    cost: float = 0.0,
    metric: str = "sharpe_ratio",
    ann_factor: float = 365.0,
) -> dict[int, float]:
    """The strategy's metric when its positions arrive ``d`` bars late, for
    each ``d`` in ``delays``: ``{d: value}``.

    ``positions[t]`` is the exposure held over ``returns[t]``, as in
    :func:`permutation_test`, ``(bars,)`` or ``(bars, assets)``; a delay of
    ``d`` holds ``positions[t - d]`` instead (nothing before). The account's
    return is the sum over assets of position times return, less ``cost``
    times the change in each position. ``metric`` is a
    :mod:`phasor.returns` metric. This is :func:`signal_delay_test` for a
    caller that has the positions rather than a ``@phasor.strategy``: an edge
    that halves at one bar of delay lives in the bar it was decided on, and
    one that grows with delay is trading too early.
    """
    from . import returns as _returns

    pos = np.nan_to_num(np.asarray(positions, dtype=float))
    ret = np.nan_to_num(np.asarray(returns, dtype=float))
    if pos.shape != ret.shape or pos.ndim not in (1, 2):
        raise ValueError(f"positions and returns must have one shape, 1-D or 2-D; got {pos.shape} and {ret.shape}")
    if pos.ndim == 1:
        pos, ret = pos[:, None], ret[:, None]
    out = {}
    for d in delays:
        d = _int("delay", d, 0)
        held = np.zeros_like(pos)
        if d < len(pos):
            held[d:] = pos[: len(pos) - d]
        traded = np.abs(np.diff(held, axis=0, prepend=0.0)).sum(axis=1)
        account = (held * ret).sum(axis=1) - cost * traded
        out[d] = float(_returns.metric(account, metric, ann_factor=ann_factor))
    return out


def signal_delay_test(
    strategy: Any,
    data: Any,
    params: Mapping[str, Any] | None = None,
    *,
    delays: Iterable[int] = (0, 1, 2),
    metric: str = "total_return",
    **options: Any,
) -> dict[int, np.ndarray]:
    """Runs a ``@phasor.strategy`` with each ``signal_delay`` and returns
    ``{delay: metric per column}``, side by side. A result that holds at
    delay 1 and collapses at delay 2 trades a one-bar effect that execution
    latency would eat; one that is much better at delay 0 than at 1 is
    probably reading its own bar's close."""
    out = {}
    for d in delays:
        pf = strategy.run(data, params=params, signal_delay=int(d), **options)
        out[int(d)] = np.asarray(getattr(pf, metric)())
    return out


def shuffled_prices(close: Any, n: int, *, block: int = 1, seed: int = 0) -> np.ndarray:
    """``n`` price paths with ``close``'s bar-to-bar log returns in a random
    order, shuffled in blocks of ``block`` bars, each path starting at the
    first close. All columns share one permutation, so the cross-section's
    correlation survives; trends, cycles and any order the strategy could
    exploit do not. Missing prices stay missing where they were.

    Returns an array ``(n, rows, columns)`` (``(n, rows)`` for 1-D input).
    """
    c = np.asarray(close, dtype=float)
    flat = c.ndim == 1
    c = c[:, None] if flat else c
    n = _int("n", n, 1)
    block = _int("block", block, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(c), axis=0)
    gaps = np.isnan(logret)
    logret = np.where(gaps, 0.0, logret)
    starts = list(range(0, len(logret), block))
    rng = np.random.default_rng(seed)
    out = np.empty((n, *c.shape))
    for k in range(n):
        order = np.concatenate([np.arange(s, min(s + block, len(logret))) for s in rng.permutation(starts)])
        path = c[0] * np.exp(np.vstack([np.zeros((1, c.shape[1])), np.cumsum(logret[order], axis=0)]))
        path[np.isnan(c)] = np.nan
        out[k] = path
    return out[:, :, 0] if flat else out


def shuffled_test(fn: Callable[[np.ndarray], float], close: Any, n: int = 100, *,
                  block: int = 1, seed: int = 0) -> dict[str, Any]:
    """Reruns ``fn(prices) -> score`` (a whole backtest, say) on
    :func:`shuffled_prices` and compares the real score with them. Returns
    ``observed``, ``null`` and ``p_value`` = ``(1 + #null >= observed) /
    (n + 1)``. A strategy that scores as well on shuffled prices is not
    using the order of returns, which is what it claims to trade."""
    observed = float(fn(np.asarray(close, dtype=float)))
    null = np.array([float(fn(path)) for path in shuffled_prices(close, n, block=block, seed=seed)])
    p = (1 + int((null >= observed).sum())) / (n + 1)
    return {"observed": observed, "null": null, "p_value": p}
