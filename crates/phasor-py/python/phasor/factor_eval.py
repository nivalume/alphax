"""How well a factor ranks the returns that follow it.

**An IC reads the future.** A forward return on bar ``t`` is a price that did
not exist when the factor was known, and an IC summarizes those returns. Use
one to decide whether a factor is worth simulating. An IC is not a signal,
and neither is a forward return.

Quantile labels are a same-bar transform of the factor. Turnover and rank
autocorrelation compare a bar with an earlier one. :func:`quantile_backtest`
is the function here that runs a simulation: it holds each quantile group.

``close`` and each factor are ``(bars,)`` or ``(bars, symbols)``. A 1-D input
is one symbol. The engine is column-major, so arrays go through
``np.asfortranarray`` — free when they already are, a copy when they are not.

    report = phasor.factor_eval.evaluate({"momentum": factor}, close)
    report.ic("momentum", h=5)          # rank IC on each bar
    report.summary()                     # mean IC, ICIR, Newey–West t
    report.decay("momentum")             # mean rank IC at each horizon
    report.quantile_returns("momentum")  # forward return of each quantile
    report.long_short("momentum")        # top quantile minus bottom

Neutralization and the correlation matrices are same-bar transforms of the
factor. Fama–MacBeth regresses a forward return, so it reads the future the
same way an IC does.

    phasor.factor_eval.neutralize(factor, groups, [beta])
    phasor.factor_eval.fama_macbeth(forward, {"momentum": factor})
    phasor.factor_eval.correlation({"momentum": factor, "beta": beta})

:func:`placebo` builds a null for the rank IC by permuting each bar or by
rotating the forward return in time. :func:`adjust_pvalues` corrects a list
of those p-values for the number of factors that were tried.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import _phasor
from ._columns import columns, restore

__all__ = [
    "FactorReport",
    "FmReport",
    "QuantileBacktestResult",
    "adjust_pvalues",
    "correlation",
    "cs_ic",
    "evaluate",
    "fama_macbeth",
    "forward_returns",
    "long_short_returns",
    "max_correlation",
    "monotonicity",
    "neutralize",
    "placebo",
    "quantile_backtest",
    "quantile_labels",
    "quantile_returns",
    "quantile_turnover",
    "rank_autocorr",
    "summarize",
    "ts_block_ic",
    "ts_ic",
    "ts_rolling_ic",
]

_SUMMARY_COLUMNS = ["ic", "rank_ic", "icir", "t", "t_nw", "p_nw", "hit", "n"]


def _count(value: int, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be 0 or more")
    return out


def _u64(value: int, name: str) -> int:
    out = int(value)
    if out < 0 or out > 2**64 - 1:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")
    return out


def _at_least(value: int, minimum: int, name: str) -> int:
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be {minimum} or more")
    return out


def _labels(value: Any, name: str = "labels") -> np.ndarray:
    """Integer labels as a 2-D F-ordered ``int16`` array. A 1-D input is one column."""
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be integers; got {array.dtype}")
    if array.size and (int(array.min()) < -1 or int(array.max()) > np.iinfo(np.int16).max):
        raise ValueError(f"{name} must fit in int16, from -1 up")
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got {array.ndim} dimensions")
    return np.asfortranarray(array.astype(np.int16, copy=False))


def _panel(value: Any, name: str) -> np.ndarray:
    """``value`` as a 2-D F-ordered float64 array. A 1-D input is one column."""
    array, _ = columns(value, name)
    return array


def _match_panel(value: Any, shape: tuple[int, int], name: str) -> np.ndarray | None:
    """``value`` as a panel of ``shape``, or ``None`` when ``value`` is ``None``."""
    if value is None:
        return None
    panel = _panel(value, name)
    if panel.shape != shape:
        raise ValueError(f"{name} has shape {panel.shape}, expected {shape}")
    return panel


def _side_panels(value: Any, shape: tuple[int, int], name: str) -> tuple[list[str], list[np.ndarray]]:
    """Panels from a mapping, a sequence, one array, or nothing.

    A mapping's keys are the names. A sequence is named ``{name}_0`` onwards.
    One array is a single panel named ``name``.
    """
    if value is None:
        return [], []
    if isinstance(value, Mapping) and not hasattr(value, "shape"):
        names: list[str] = []
        panels: list[np.ndarray] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} names must be strings")
            panel = _panel(item, f"{name} {key!r}")
            if panel.shape != shape:
                raise ValueError(f"{name} {key!r} has shape {panel.shape}, expected {shape}")
            names.append(key)
            panels.append(panel)
        return names, panels
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        panels = []
        for i, item in enumerate(value):
            panel = _panel(item, f"{name}[{i}]")
            if panel.shape != shape:
                raise ValueError(f"{name}[{i}] has shape {panel.shape}, expected {shape}")
            panels.append(panel)
        return [f"{name}_{i}" for i in range(len(panels))], panels
    panel = _panel(value, name)
    if panel.shape != shape:
        raise ValueError(f"{name} has shape {panel.shape}, expected {shape}")
    return [name], [panel]


def _method(name: str) -> str:
    if name == "pearson":
        return "pearson"
    if name in ("spearman", "rank"):
        return "spearman"
    raise ValueError('method must be "pearson", "spearman" or "rank"')


def _horizons(horizons: Sequence[int]) -> list[int]:
    out = [int(h) for h in horizons]
    if not out:
        raise ValueError("horizons cannot be empty")
    if any(h < 1 for h in out):
        raise ValueError("each horizon must be 1 or more")
    if len(set(out)) != len(out):
        raise ValueError("horizons must be unique")
    return out


def forward_returns(close: Any, horizons: Sequence[int] = (1, 5, 10, 20), lag: int = 1) -> dict[int, np.ndarray]:
    """``close[t + lag + horizon] / close[t + lag] - 1`` for each horizon.

    ``lag`` is the delay between knowing the factor and trading on it: with
    the default ``lag=1``, a factor stamped on bar ``t`` is scored from the
    next bar. ``lag=0`` is the return from the factor's own bar. The result
    is missing where the window runs past the symbol or a close is not
    finite. Looks ``lag + horizon`` bars ahead.

    Returns a dict keyed by horizon, each value shaped like ``close``.
    """
    arr, one_d = columns(close, "close")
    delay = _count(lag, "lag")
    out: dict[int, np.ndarray] = {}
    for horizon in _horizons(horizons):
        out[horizon] = restore(_phasor.factor_forward_returns(arr, horizon, delay), one_d)
    return out


def cs_ic(
    factors: Sequence[Any],
    forwards: Sequence[Any],
    min_names: int = 10,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional Pearson and Spearman IC, per bar.

    ``factors`` and ``forwards`` are sequences of panels shaped
    ``(bars, symbols)``. Returns ``(pearson, spearman)``, each
    ``(n_factors, n_horizons, bars)``. A bar with fewer than ``min_names``
    symbols where both sides are finite, or a constant side, is NaN.
    ``threads=0`` uses every core; the values do not depend on how many.
    """
    return _phasor.factor_cs_ic(
        [_panel(factor, "factor") for factor in factors],
        [_panel(forward, "forward") for forward in forwards],
        _count(min_names, "min_names"),
        _count(threads, "threads"),
    )


def ts_ic(
    factor: Any,
    forward: Any,
    method: str = "pearson",
    min_periods: int = 2,
    threads: int = 0,
) -> np.ndarray:
    """Time-series IC of each symbol over the whole sample.

    One value per symbol: the correlation of that symbol's factor and
    forward return on the bars where both are finite. NaN below
    ``min_periods`` bars or when a side is constant. ``method`` is
    ``"pearson"``, ``"spearman"`` or ``"rank"``.
    """
    return _phasor.factor_ts_ic(
        _panel(factor, "factor"),
        _panel(forward, "forward"),
        method,
        _count(min_periods, "min_periods"),
        _count(threads, "threads"),
    )


def ts_rolling_ic(
    factor: Any,
    forward: Any,
    window: int,
    min_periods: int = 2,
    method: str = "pearson",
    threads: int = 0,
) -> np.ndarray:
    """Time-series IC of each symbol over a trailing window, shaped like ``factor``.

    On row ``t`` the window is the last ``window`` bars ending at ``t``.
    The forward return inside that window is only known later, so the result
    describes the factor and is not a signal. ``method`` is ``"pearson"``,
    ``"spearman"`` or ``"rank"``.
    """
    arr, one_d = columns(factor, "factor")
    fwd = _panel(forward, "forward")
    if fwd.shape != arr.shape:
        raise ValueError(f"forward has shape {fwd.shape}, expected {arr.shape}")
    out = _phasor.factor_ts_rolling_ic(
        arr,
        fwd,
        _count(window, "window"),
        _count(min_periods, "min_periods"),
        method,
        _count(threads, "threads"),
    )
    return restore(out, one_d)


def ts_block_ic(
    factor: Any,
    forward: Any,
    blocks: Any,
    min_periods: int = 2,
    method: str = "pearson",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-series IC of each symbol inside each block of rows.

    ``blocks[t]`` names row ``t``'s block, for example a month id. Rows of a
    block need not be adjacent. Returns ``(ids, ic)``: the distinct ids in
    ascending order, and the IC of each block. ``ic`` is
    ``(n_blocks, symbols)``, or ``(n_blocks,)`` when ``factor`` was 1-D.
    ``method`` is ``"pearson"``, ``"spearman"`` or ``"rank"``.
    """
    arr, one_d = columns(factor, "factor")
    fwd = _panel(forward, "forward")
    if fwd.shape != arr.shape:
        raise ValueError(f"forward has shape {fwd.shape}, expected {arr.shape}")
    ids = np.asarray(blocks)
    if ids.dtype.kind not in "iu":
        raise TypeError(f"blocks must be integer ids; got {ids.dtype}")
    ids = np.ascontiguousarray(ids.reshape(-1).astype(np.int64, copy=False))
    if ids.shape[0] != arr.shape[0]:
        raise ValueError(f"blocks has {ids.shape[0]} ids, expected one per row ({arr.shape[0]})")
    unique, ic = _phasor.factor_ts_block_ic(
        arr,
        fwd,
        ids,
        _count(min_periods, "min_periods"),
        method,
        _count(threads, "threads"),
    )
    return unique, restore(ic, one_d)


def summarize(series: Any, lags: int | None = None, horizon: int = 1) -> dict[str, float]:
    """Mean, spread and significance of one IC series.

    NaN entries are dropped first; the rest are treated as consecutive, which
    is what a Newey–West lag assumes. When ``lags`` is omitted it is the
    automatic lag for ``horizon``-bar overlapping returns: at least
    ``horizon - 1``, and at least the rule of thumb ``4 (n/100) ** (2/9)``,
    where ``n`` counts finite ICs.

    Returns ``mean``, ``std`` (sample, ``ddof=1``), ``icir`` (``mean / std``),
    ``t`` (``icir * sqrt(n)``), ``t_nw`` and ``p_nw`` (Newey–West t and its
    two-sided normal p-value), ``hit`` (share strictly above zero) and ``n``.
    """
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if lags is None:
        valid = int(np.isfinite(values).sum())
        used = int(_phasor.factor_auto_lags(valid, _count(horizon, "horizon")))
    else:
        used = _count(lags, "lags")
    mean, std, icir, t, t_nw, p_nw, hit, n = _phasor.factor_summarize(np.ascontiguousarray(values), used)
    return {
        "mean": float(mean),
        "std": float(std),
        "icir": float(icir),
        "t": float(t),
        "t_nw": float(t_nw),
        "p_nw": float(p_nw),
        "hit": float(hit),
        "n": int(n),
    }


def quantile_labels(factor: Any, q: int = 5) -> np.ndarray:
    """Quantile group of each factor value, shaped like ``factor``.

    Group ``0`` holds the smallest values and group ``q - 1`` the largest.
    On each bar the finite values are average-ranked and value with rank
    ``r`` (1-based, among ``n`` finite values) goes to
    ``floor((r - 0.5) * q / n)``. A missing value is ``-1``. A bar with fewer
    than ``q`` finite values is all ``-1``. The dtype is ``int16``.
    """
    arr, one_d = columns(factor, "factor")
    out = _phasor.factor_quantile_labels(arr, _at_least(q, 1, "q"))
    return restore(out, one_d)


def quantile_returns(labels: Any, forward: Any, q: int = 5, weights: Any = None) -> np.ndarray:
    """Mean forward return of each quantile group, shaped ``(bars, q)``.

    A name counts when its label is the group and its forward return is
    finite. With ``weights`` the mean is weighted by that panel (same shape
    as ``forward``); a non-finite weight is skipped, and a group whose
    weights do not sum to a positive number is NaN. Without weights every
    counted name has an equal weight. Column ``g`` is group ``g``.
    """
    lab = _labels(labels)
    fwd = _panel(forward, "forward")
    if lab.shape != fwd.shape:
        raise ValueError(f"labels have shape {lab.shape}, expected {fwd.shape}")
    groups = _at_least(q, 1, "q")
    if weights is None:
        weighted = None
    else:
        weighted = _panel(weights, "weights")
        if weighted.shape != fwd.shape:
            raise ValueError(f"weights have shape {weighted.shape}, expected {fwd.shape}")
    return np.asarray(_phasor.factor_quantile_returns(lab, fwd, groups, weighted))


def long_short_returns(q_returns: Any, q: int = 5) -> np.ndarray:
    """Top quantile minus bottom quantile, one value per bar.

    ``q_returns`` is ``(bars, q)``, as :func:`quantile_returns` returns it.
    Entry ``t`` is column ``q - 1`` minus column ``0``.
    """
    groups = _at_least(q, 1, "q")
    arr = np.asarray(q_returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"q_returns must be 1-D or 2-D; got {arr.ndim} dimensions")
    if arr.shape[1] != groups:
        raise ValueError(f"q_returns has {arr.shape[1]} groups, expected {groups}")
    return np.asarray(_phasor.factor_long_short_returns(np.asfortranarray(arr), groups))


def monotonicity(mean_q_returns: Any) -> float:
    """Spearman correlation of ``0 .. q-1`` with each group's mean return.

    NaN below two groups, when a mean is not finite, or when every mean is
    the same number.
    """
    values = np.ascontiguousarray(np.asarray(mean_q_returns, dtype=np.float64).reshape(-1))
    return float(_phasor.factor_monotonicity(values))


def quantile_turnover(labels: Any, q: int = 5, lag: int = 1) -> np.ndarray:
    """Turnover of each quantile group, shaped ``(bars, q)``.

    On bar ``t`` group ``g``'s turnover is ``1 - |S_t ∩ S_{t-lag}| / |S_t|``,
    where ``S`` is the set of symbols labelled ``g``. An empty group is NaN.
    Bars before ``lag`` are NaN. ``0`` means the group held the same names.
    """
    lab = _labels(labels)
    return np.asarray(
        _phasor.factor_quantile_turnover(lab, _at_least(q, 1, "q"), _count(lag, "lag"))
    )


def rank_autocorr(factor: Any, lag: int = 1) -> np.ndarray:
    """Spearman correlation of each bar's factor with the bar ``lag`` before it.

    One value per bar. Symbols where either bar is missing are dropped. NaN
    before ``lag``, below two symbols in common, or when either side is constant.
    """
    return np.asarray(_phasor.factor_rank_autocorr(_panel(factor, "factor"), _count(lag, "lag")))


def neutralize(
    y: Any,
    groups: Any = None,
    exposures: Sequence[Any] | Mapping[str, Any] | None = None,
    threads: int = 0,
) -> np.ndarray:
    """Residuals of ``y`` after its group means and its exposures.

    On each bar the symbols where ``y``, ``groups`` and every exposure are
    finite are kept. ``groups``, when given, is a panel of discrete labels of
    the same shape as ``y``; ``y`` and the exposures are demeaned inside each
    label. Exposures are then regressed out, with an intercept. A constant
    exposure, or an exposure of zeros, is absorbed by that intercept, so the
    result is ``y`` with its cross-sectional mean removed. A bar that cannot
    identify the slopes — too few symbols, or exposures that are linear
    copies of each other — is ``NaN``.

    ``exposures`` is a sequence of panels, a mapping of names to panels, or
    one panel. The result has ``y``'s shape, including a 1-D input coming
    back 1-D. This is a same-bar transform: it does not read the future, and
    the residual may be used as a signal. ``threads=0`` uses every core; the
    values do not depend on how many.
    """
    arr, one_d = columns(y, "y")
    group = _match_panel(groups, arr.shape, "groups")
    _, exposure_panels = _side_panels(exposures, arr.shape, "exposure")
    out = _phasor.factor_neutralize(arr, group, exposure_panels, _count(threads, "threads"))
    return restore(np.asarray(out), one_d)


_FM_COLUMNS = ["mean", "std", "t_nw", "p_nw"]


class FmReport:
    """Fama–MacBeth coefficients, one cross-sectional regression per bar.

    ``coef`` is the coefficient on each bar: an intercept, then each factor,
    then each control. ``r2`` is that bar's R². :meth:`summary` is the
    time-series mean of each coefficient (the risk premium), its sample
    standard deviation, and a Newey–West t-statistic with the automatic lag
    for non-overlapping bars. Bars where the cross-section was singular are
    ``NaN`` and are dropped from the summary.

    A :class:`pandas.DataFrame` and :class:`pandas.Series` when pandas is
    installed. Otherwise ``coef`` is a dict of name to array and ``r2`` is
    an array.
    """

    def __init__(self, names: Sequence[str], coef: np.ndarray, r2: np.ndarray) -> None:
        self.names = list(names)
        coef = np.asarray(coef, dtype=np.float64)
        r2 = np.asarray(r2, dtype=np.float64)
        self._r2 = r2
        try:
            import pandas as pd
        except ImportError:
            self.coef: Any = {name: coef[:, i].copy() for i, name in enumerate(self.names)}
            self.r2: Any = r2
        else:
            self.coef = pd.DataFrame(coef, columns=self.names)
            self.r2 = pd.Series(r2, name="r2")

    def summary(self) -> Any:
        """Mean risk premium, standard deviation, Newey–West t and its p-value.

        One row per coefficient. A DataFrame indexed by name when pandas is
        installed, otherwise a dict of name to that row.
        """
        rows = []
        for name in self.names:
            stats = summarize(np.asarray(self.coef[name], dtype=np.float64).reshape(-1), horizon=1)
            rows.append(
                {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "t_nw": stats["t_nw"],
                    "p_nw": stats["p_nw"],
                }
            )
        try:
            import pandas as pd
        except ImportError:
            return {name: row for name, row in zip(self.names, rows)}
        frame = pd.DataFrame(rows, columns=_FM_COLUMNS)
        frame.index = self.names
        frame.index.name = "factor"
        return frame

    def __repr__(self) -> str:
        return f"FmReport(names={self.names!r}, rows={int(self._r2.shape[0])})"


def fama_macbeth(
    forward: Any,
    factors: Mapping[str, Any],
    groups: Any = None,
    controls: Sequence[Any] | Mapping[str, Any] | None = None,
    threads: int = 0,
) -> FmReport:
    """Cross-sectional regression of ``forward`` on the factors, one bar at a time.

    On each bar the regression is ``forward ~ 1 + factors + controls``, over
    the symbols that are finite on all of them. ``groups``, when given, adds
    a fixed effect per discrete label, removed by demeaning inside the group;
    the reported intercept is then the average of those group intercepts.
    A bar with too few symbols, or with regressors that do not vary, is
    ``NaN``.

    ``factors`` is a mapping of names to panels. ``controls`` is a mapping, a
    sequence of panels, or one panel. Control coefficients are estimated and
    reported; the risk premia are the factor columns. :meth:`FmReport.summary`
    is the time-series mean of each coefficient, with a Newey–West t.

    ``forward`` is usually a forward return, which reads the future. These
    coefficients describe the factors. They are not a signal.
    ``threads=0`` uses every core; the values do not depend on how many.
    """
    if not isinstance(factors, Mapping) or hasattr(factors, "shape"):
        raise TypeError("factors must be a mapping of names to panels")
    names, panels = _factor_panels(factors)
    shape = panels[0].shape
    fwd = _match_panel(forward, shape, "forward")
    if fwd is None:
        raise ValueError("forward is required")
    group = _match_panel(groups, shape, "groups")
    control_names, control_panels = _side_panels(controls, shape, "control")
    rust_names, coef, r2 = _phasor.factor_fama_macbeth(
        fwd,
        panels,
        names,
        group,
        control_panels,
        _count(threads, "threads"),
    )
    out_names = list(rust_names)
    if control_names and isinstance(controls, Mapping) and not hasattr(controls, "shape"):
        out_names = out_names[: len(out_names) - len(control_names)] + control_names
    return FmReport(out_names, np.asarray(coef), np.asarray(r2))


def correlation(factors: Mapping[str, Any], mode: str = "cs", threads: int = 0) -> Any:
    """Mean cross-sectional Spearman correlation of every factor with every other.

    ``mode`` is ``"cs"``: on each bar the factors are rank-correlated across
    symbols, and those correlations are averaged over the bars where both
    factors had at least two varying finite values. The diagonal is 1 where
    a factor varies on at least one bar, and ``NaN`` when it never does.

    A DataFrame indexed by factor name when pandas is installed, otherwise a
    dict of dicts. ``threads=0`` uses every core; the values do not depend
    on how many.
    """
    if mode != "cs":
        raise ValueError('mode must be "cs"')
    names, panels = _factor_panels(factors)
    matrix = np.asarray(_phasor.factor_mean_cs_corr(panels, _count(threads, "threads")))
    try:
        import pandas as pd
    except ImportError:
        return {
            name: {other: float(matrix[i, j]) for j, other in enumerate(names)}
            for i, name in enumerate(names)
        }
    frame = pd.DataFrame(matrix, index=names, columns=names)
    frame.index.name = "factor"
    frame.columns.name = "factor"
    return frame


def max_correlation(candidates: Mapping[str, Any], references: Mapping[str, Any], threads: int = 0) -> Any:
    """For each candidate, the reference it is most correlated with.

    The correlation is the mean cross-sectional Spearman correlation from
    :func:`correlation`, and the score is its absolute value, so an exact
    reverse of a reference scores 1 and names that reference. Ties keep the
    reference that was given first. A candidate with no finite correlation
    has a ``NaN`` score and no closest name.

    A DataFrame with columns ``max_abs_corr`` and ``closest`` when pandas is
    installed, otherwise a dict of candidate name to those two fields.
    """
    cand_names, cand_panels = _factor_panels(candidates)
    ref_names, ref_panels = _factor_panels(references)
    if cand_panels[0].shape != ref_panels[0].shape:
        raise ValueError(
            f"references have shape {ref_panels[0].shape}, expected {cand_panels[0].shape}"
        )
    matrix = np.asarray(
        _phasor.factor_cross_corr(cand_panels, ref_panels, _count(threads, "threads"))
    )
    max_abs: list[float] = []
    closest: list[Any] = []
    for i in range(len(cand_names)):
        best_j = None
        best = -1.0
        for j, value in enumerate(np.asarray(matrix[i]).reshape(-1)):
            if not np.isfinite(value):
                continue
            mag = abs(float(value))
            if best_j is None or mag > best:
                best = mag
                best_j = j
        if best_j is None:
            max_abs.append(float("nan"))
            closest.append(None)
        else:
            max_abs.append(best)
            closest.append(ref_names[best_j])
    try:
        import pandas as pd
    except ImportError:
        return {
            name: {"max_abs_corr": max_abs[i], "closest": closest[i]}
            for i, name in enumerate(cand_names)
        }
    frame = pd.DataFrame({"max_abs_corr": max_abs, "closest": closest}, index=cand_names)
    frame.index.name = "candidate"
    return frame


def placebo(
    factor: Any,
    forward: Any,
    draws: int = 500,
    seed: int = 0,
    mode: str = "cs",
    min_names: int = 10,
    threads: int = 0,
    min_periods: int = 2,
) -> dict[str, Any]:
    """Rank IC of ``factor`` against ``forward``, and a p-value from a placebo null.

    ``mode="cs"`` permutes the symbols' forward returns on each bar and
    averages the per-bar Spearman IC. ``min_names`` is the smallest number of
    finite pairs a bar needs. ``mode="ts"`` circularly shifts ``forward``
    along time by an offset in ``[n / 10, n - n / 10]`` and averages the
    per-symbol Spearman IC. ``min_periods`` is the smallest number of finite
    pairs a symbol needs; it is ignored for ``mode="cs"``.

    ``draws`` nulls are built. The p-value is
    ``(1 + #{|null| >= |observed|}) / (1 + draws)``, so the observed result
    counts as one more draw under the null. ``seed`` selects the null; the
    same seed and the same panels give the same null on any number of
    threads. ``threads=0`` uses every core.

    Returns ``observed``, ``p_value`` and ``null_distribution`` (one mean
    rank IC per draw). ``p_value`` is ``NaN`` when the observed rank IC is.
    Both panels read whatever future the caller put in them: a forward
    return does, and this p-value describes the factor. It is not a signal.
    """
    panel = _panel(factor, "factor")
    fwd = _panel(forward, "forward")
    if fwd.shape != panel.shape:
        raise ValueError(f"forward has shape {fwd.shape}, expected {panel.shape}")
    n_draws = _count(draws, "draws")
    n_threads = _count(threads, "threads")
    chosen = _u64(seed, "seed")
    if mode == "cs":
        observed, p_value, null = _phasor.factor_cs_permutation(
            panel,
            fwd,
            n_draws,
            chosen,
            _count(min_names, "min_names"),
            n_threads,
        )
    elif mode == "ts":
        observed, p_value, null = _phasor.factor_ts_circular_shift(
            panel,
            fwd,
            n_draws,
            chosen,
            _count(min_periods, "min_periods"),
            n_threads,
        )
    else:
        raise ValueError('mode must be "cs" or "ts"')
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null_distribution": np.asarray(null, dtype=np.float64),
    }


def adjust_pvalues(pvalues: Any, method: str = "bh") -> np.ndarray:
    """P-values corrected for the number of tests, the same shape as ``pvalues``.

    ``m`` is the number of finite p-values. A non-finite input stays ``NaN``
    and is not a test.

    ``"bonferroni"`` is ``min(1, p * m)``. ``"holm"`` is the step-down
    Bonferroni: sort ascending, multiply ``p_(i)`` by ``m - i + 1``, then
    take the cumulative maximum and cap at 1. ``"bh"`` and ``"fdr"`` are the
    Benjamini–Hochberg false-discovery rate: multiply ``p_(i)`` by ``m / i``,
    then take the cumulative minimum from the largest and cap at 1. Equal
    p-values keep their original order.
    """
    values = np.asarray(pvalues, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    flat = values.reshape(-1)
    finite = np.isfinite(flat)
    m = int(finite.sum())
    if m == 0:
        return out
    p = flat[finite]
    if method == "bonferroni":
        adjusted = np.clip(p * m, 0.0, 1.0)
    elif method == "holm":
        order = np.argsort(p, kind="mergesort")
        running = 0.0
        stepped = np.empty(m, dtype=np.float64)
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * float(p[idx]))
            stepped[rank] = running
        adjusted = np.empty(m, dtype=np.float64)
        adjusted[order] = np.minimum(stepped, 1.0)
    elif method in ("bh", "fdr"):
        order = np.argsort(p, kind="mergesort")
        ranked = p[order]
        stepped = np.empty(m, dtype=np.float64)
        running = 1.0
        for rank in range(m - 1, -1, -1):
            raw = float(ranked[rank]) * m / (rank + 1)
            running = raw if raw < running else running
            stepped[rank] = running if running < 1.0 else 1.0
        adjusted = np.empty(m, dtype=np.float64)
        adjusted[order] = stepped
    else:
        raise ValueError('method must be "bonferroni", "holm", "bh" or "fdr"')
    out.reshape(-1)[finite] = adjusted
    return out


def _column_means(values: np.ndarray) -> np.ndarray:
    means = np.empty(values.shape[1], dtype=np.float64)
    for g in range(values.shape[1]):
        finite = values[:, g]
        finite = finite[np.isfinite(finite)]
        means[g] = float(finite.mean()) if finite.size else np.nan
    return means


class FactorReport:
    """Cross-sectional ICs of several factors at several horizons.

    ``pearson`` and ``spearman`` are ``(n_factors, n_horizons, bars)``.
    ``summaries`` maps ``(factor, horizon, "pearson" | "spearman")`` to the
    dict :func:`summarize` returns. ``index``, when given to :func:`evaluate`,
    is kept so a caller can align :meth:`ic` with the bars it came from.

    ``factors`` and ``forwards`` are the panels :func:`evaluate` scored.
    :meth:`quantile_returns`, :meth:`turnover` and :meth:`rank_autocorr` read
    them. A report built without those panels can still answer :meth:`ic`.
    """

    def __init__(
        self,
        names: Sequence[str],
        horizons: Sequence[int],
        pearson: np.ndarray,
        spearman: np.ndarray,
        summaries: dict[tuple[str, int, str], dict[str, float]],
        index: Any = None,
        factors: Mapping[str, np.ndarray] | None = None,
        forwards: Mapping[int, np.ndarray] | None = None,
        neutral_rank_ic: Mapping[tuple[str, int], float] | None = None,
    ) -> None:
        self.names = list(names)
        self.horizons = [int(h) for h in horizons]
        self.pearson = np.asarray(pearson, dtype=np.float64)
        self.spearman = np.asarray(spearman, dtype=np.float64)
        self.summaries = summaries
        self.neutral_rank_ic = None if neutral_rank_ic is None else dict(neutral_rank_ic)
        self.index = index
        self.factors = {} if factors is None else {str(name): np.asarray(panel) for name, panel in factors.items()}
        self.forwards = {} if forwards is None else {int(horizon): np.asarray(panel) for horizon, panel in forwards.items()}

    def ic(self, factor: str, h: int = 1, method: str = "rank") -> np.ndarray:
        """The IC of ``factor`` at horizon ``h`` on each bar.

        ``method="rank"`` (and ``"spearman"``) is the rank IC; ``"pearson"``
        is the Pearson IC. Missing bars stay NaN.
        """
        cube = self.spearman if _method(method) == "spearman" else self.pearson
        return np.array(cube[self._factor(factor), self._horizon(h)], dtype=np.float64, copy=True)

    def summary(self) -> Any:
        """One row per ``(factor, horizon)``.

        ``ic`` is the mean Pearson IC and ``rank_ic`` the mean Spearman IC.
        ``icir``, ``t``, ``t_nw``, ``p_nw``, ``hit`` and ``n`` describe the
        rank IC, which is the series :meth:`ic` returns by default. The two
        series are missing on the same bars, so ``n`` is the same either way.
        When :func:`evaluate` was given ``groups`` or ``exposures``,
        ``neutral_rank_ic`` is the mean rank IC of the factor after those
        were removed.

        A :class:`pandas.DataFrame` with that multi-index when pandas is
        installed, otherwise a dict keyed by ``(factor, horizon)``.
        """
        columns = list(_SUMMARY_COLUMNS)
        if self.neutral_rank_ic is not None:
            columns.append("neutral_rank_ic")
        rows = []
        keys = []
        for name in self.names:
            for horizon in self.horizons:
                pearson = self.summaries[(name, horizon, "pearson")]
                spearman = self.summaries[(name, horizon, "spearman")]
                row = {
                    "ic": pearson["mean"],
                    "rank_ic": spearman["mean"],
                    "icir": spearman["icir"],
                    "t": spearman["t"],
                    "t_nw": spearman["t_nw"],
                    "p_nw": spearman["p_nw"],
                    "hit": spearman["hit"],
                    "n": spearman["n"],
                }
                if self.neutral_rank_ic is not None:
                    row["neutral_rank_ic"] = self.neutral_rank_ic[(name, horizon)]
                rows.append(row)
                keys.append((name, horizon))
        try:
            import pandas as pd
        except ImportError:
            return dict(zip(keys, rows))
        frame = pd.DataFrame(rows, columns=columns)
        frame.index = pd.MultiIndex.from_tuples(keys, names=["factor", "horizon"])
        return frame

    def decay(self, factor: str, method: str = "rank") -> Any:
        """Mean IC of ``factor`` at each horizon, in horizon order.

        A :class:`pandas.Series` indexed by horizon when pandas is installed,
        otherwise a dict.
        """
        which = _method(method)
        means = {horizon: self.summaries[(factor, horizon, which)]["mean"] for horizon in self.horizons}
        try:
            import pandas as pd
        except ImportError:
            return means
        return pd.Series(means, name=factor, dtype=float)

    def quantile_returns(self, factor: str, h: int = 1, q: int = 5, weights: Any = None) -> Any:
        """Forward return of each quantile of ``factor`` at horizon ``h``.

        A :class:`pandas.DataFrame` with columns ``0 .. q-1`` and this
        report's ``index`` when pandas is installed, otherwise the
        ``(bars, q)`` array.
        """
        return self._table(self._quantile_matrix(factor, h, q, weights))

    def long_short(self, factor: str, h: int = 1, q: int = 5) -> np.ndarray:
        """Top quantile minus bottom quantile of ``factor`` at horizon ``h``."""
        return long_short_returns(self._quantile_matrix(factor, h, q, None), q)

    def monotonicity(self, factor: str, h: int = 1, q: int = 5) -> float:
        """Rank correlation of the quantile index with its mean forward return."""
        return monotonicity(_column_means(self._quantile_matrix(factor, h, q, None)))

    def turnover(self, factor: str, q: int = 5, lag: int = 1) -> Any:
        """Turnover of each quantile of ``factor``, a ``(bars, q)`` table.

        Same table shape as :meth:`quantile_returns`: a DataFrame when pandas
        is installed, otherwise the array.
        """
        labels = quantile_labels(self._factor_panel(factor), q)
        return self._table(quantile_turnover(labels, q, lag))

    def rank_autocorr(self, factor: str, lag: int = 1) -> np.ndarray:
        """Rank autocorrelation of ``factor`` at ``lag``, one value per bar."""
        return rank_autocorr(self._factor_panel(factor), lag)

    def _quantile_matrix(self, factor: str, horizon: int, q: int, weights: Any) -> np.ndarray:
        labels = quantile_labels(self._factor_panel(factor), q)
        return quantile_returns(labels, self._forward_panel(horizon), q, weights)

    def _factor_panel(self, factor: str) -> np.ndarray:
        self._factor(factor)
        panel = self.factors.get(factor)
        if panel is None:
            raise ValueError(f"report has no values for factor {factor!r}; build it with evaluate")
        return panel

    def _forward_panel(self, horizon: int) -> np.ndarray:
        horizon = int(horizon)
        self._horizon(horizon)
        panel = self.forwards.get(horizon)
        if panel is None:
            raise ValueError(f"report has no forward return for horizon {horizon}; build it with evaluate")
        return panel

    def _table(self, values: np.ndarray) -> Any:
        try:
            import pandas as pd
        except ImportError:
            return values
        frame = pd.DataFrame(np.asarray(values), columns=list(range(values.shape[1])))
        if self.index is not None:
            frame.index = self.index
        return frame

    def _factor(self, factor: str) -> int:
        try:
            return self.names.index(factor)
        except ValueError:
            raise ValueError(f"no factor {factor!r}; have {self.names}") from None

    def _horizon(self, horizon: int) -> int:
        horizon = int(horizon)
        try:
            return self.horizons.index(horizon)
        except ValueError:
            raise ValueError(f"no horizon {horizon}; have {self.horizons}") from None

    def __repr__(self) -> str:
        rows = int(self.pearson.shape[-1]) if self.pearson.ndim == 3 else 0
        return f"FactorReport(factors={self.names!r}, horizons={self.horizons!r}, rows={rows})"


def _factor_panels(factors: Mapping[str, Any] | Any) -> tuple[list[str], list[np.ndarray]]:
    # A DataFrame is a mapping and has a shape; it is one panel, not one
    # factor per column. A dict of panels has no shape.
    if isinstance(factors, Mapping) and not hasattr(factors, "shape"):
        if len(factors) == 0:
            raise ValueError("factors cannot be empty")
        names: list[str] = []
        panels: list[np.ndarray] = []
        shape: tuple[int, int] | None = None
        for name, value in factors.items():
            if not isinstance(name, str):
                raise TypeError("factor names must be strings")
            panel = _panel(value, f"factor {name!r}")
            if shape is None:
                shape = panel.shape
            elif panel.shape != shape:
                raise ValueError(f"factor {name!r} has shape {panel.shape}, expected {shape}")
            names.append(name)
            panels.append(panel)
        return names, panels
    return ["factor"], [_panel(factors, "factor")]


def evaluate(
    factors: Mapping[str, Any] | Any,
    close: Any = None,
    *,
    forward: Mapping[int, Any] | None = None,
    horizons: Sequence[int] = (1, 5, 10, 20),
    lag: int = 1,
    min_names: int = 10,
    threads: int = 0,
    index: Any = None,
    groups: Any = None,
    exposures: Sequence[Any] | Mapping[str, Any] | None = None,
) -> FactorReport:
    """Cross-sectional IC of each factor against each horizon's forward return.

    ``factors`` is a dict of names to ``(bars, symbols)`` panels, or one panel
    (named ``"factor"``). Pass ``close`` and the forward returns are built
    from it, or pass ``forward`` as a dict of horizon to a panel already
    built that way — one of the two, not both.

    Both the Pearson and the rank IC are scored, and each series is
    summarized with the Newey–West lag its horizon calls for. ``index``, when
    given, must have one entry per bar and is stored on the report.

    ``groups`` and ``exposures`` are handed to :func:`neutralize`. When either
    is given, :meth:`FactorReport.summary` gains a ``neutral_rank_ic`` column:
    the mean rank IC of the factor after those were removed. The plain
    ``rank_ic`` is still the factor as it was passed in.
    """
    if close is None and forward is None:
        raise ValueError("pass close or forward")
    if close is not None and forward is not None:
        raise ValueError("pass only one of close or forward")
    names, panels = _factor_panels(factors)
    shape = panels[0].shape
    wanted = _horizons(horizons)
    if forward is None:
        close_arr = _panel(close, "close")
        if close_arr.shape != shape:
            raise ValueError(f"close has shape {close_arr.shape}, expected {shape}")
        forwards = [_phasor.factor_forward_returns(close_arr, horizon, _count(lag, "lag")) for horizon in wanted]
    else:
        provided = {int(horizon): value for horizon, value in forward.items()}
        forwards = []
        for horizon in wanted:
            if horizon not in provided:
                raise ValueError(f"forward has no horizon {horizon}")
            panel = _panel(provided[horizon], f"forward horizon {horizon}")
            if panel.shape != shape:
                raise ValueError(f"forward horizon {horizon} has shape {panel.shape}, expected {shape}")
            forwards.append(panel)
    if index is not None and len(index) != shape[0]:
        raise ValueError(f"index has length {len(index)}, expected {shape[0]} rows")
    min_obs = _count(min_names, "min_names")
    n_threads = _count(threads, "threads")
    pearson, spearman = _phasor.factor_cs_ic(panels, forwards, min_obs, n_threads)
    pearson = np.asarray(pearson)
    spearman = np.asarray(spearman)
    summaries: dict[tuple[str, int, str], dict[str, float]] = {}
    for i, name in enumerate(names):
        for j, horizon in enumerate(wanted):
            summaries[(name, horizon, "pearson")] = summarize(pearson[i, j], horizon=horizon)
            summaries[(name, horizon, "spearman")] = summarize(spearman[i, j], horizon=horizon)
    neutral_rank_ic = None
    if groups is not None or exposures is not None:
        neutral_panels = [
            neutralize(panel, groups, exposures, threads=n_threads) for panel in panels
        ]
        _, spearman_n = cs_ic(neutral_panels, forwards, min_names=min_obs, threads=n_threads)
        spearman_n = np.asarray(spearman_n)
        neutral_rank_ic = {}
        for i, name in enumerate(names):
            for j, horizon in enumerate(wanted):
                neutral_rank_ic[(name, horizon)] = summarize(spearman_n[i, j], horizon=horizon)["mean"]
    return FactorReport(
        names,
        wanted,
        pearson,
        spearman,
        summaries,
        index,
        factors={name: panel for name, panel in zip(names, panels)},
        forwards={horizon: panel for horizon, panel in zip(wanted, forwards)},
        neutral_rank_ic=neutral_rank_ic,
    )


def _group_targets(mask: np.ndarray, scale: float, active: np.ndarray) -> np.ndarray:
    """Equal weight ``scale / count`` on the names in ``mask``, elsewhere 0.

    Rows that are not ``active`` stay NaN, so the book holds through them.
    """
    count = mask.sum(axis=1)
    per = np.zeros(mask.shape[0], dtype=np.float64)
    np.divide(scale, count, out=per, where=count > 0)
    weights = np.asarray(mask * per[:, None], dtype=np.float64)
    weights[~active] = np.nan
    return weights


class QuantileBacktestResult:
    """A quantile-group backtest.

    ``portfolio`` is the :class:`phasor.Portfolio`. Groups ``0 .. quantiles-1``
    are long-only books of those quantiles. When ``long_short`` is set, the
    last group is long the top quantile and short the bottom, half the book
    each side. :meth:`values` is ``(bars, groups)`` and :meth:`stats` is one
    row per group.
    """

    def __init__(self, portfolio: Any, n_groups: int, quantiles: int, long_short: bool) -> None:
        self.portfolio = portfolio
        self.n_groups = int(n_groups)
        self.quantiles = int(quantiles)
        self.long_short = bool(long_short)

    def values(self) -> np.ndarray:
        """Equity of each group, shaped ``(bars, groups)``."""
        cols = [
            np.asarray(self.portfolio.value(group=group), dtype=np.float64)
            for group in range(self.n_groups)
        ]
        return np.column_stack(cols) if cols else np.empty((0, 0), dtype=np.float64)

    def stats(self) -> Any:
        """``total_return``, ``sharpe_ratio`` and ``max_drawdown`` of each group.

        A DataFrame indexed by group when pandas is installed, otherwise a
        dict keyed by group number.
        """
        portfolio = self.portfolio
        total = np.asarray(portfolio.total_return(), dtype=np.float64).reshape(-1)
        sharpe = np.asarray(portfolio.sharpe_ratio(), dtype=np.float64).reshape(-1)
        drawdown = np.asarray(portfolio.max_drawdown(), dtype=np.float64).reshape(-1)
        rows = [
            {
                "total_return": float(total[group]),
                "sharpe_ratio": float(sharpe[group]),
                "max_drawdown": float(drawdown[group]),
            }
            for group in range(self.n_groups)
        ]
        try:
            import pandas as pd
        except ImportError:
            return {group: row for group, row in enumerate(rows)}
        frame = pd.DataFrame(rows)
        frame.index = list(range(self.n_groups))
        frame.index.name = "group"
        return frame

    def __repr__(self) -> str:
        return (
            f"QuantileBacktestResult(groups={self.n_groups}, quantiles={self.quantiles}, "
            f"long_short={self.long_short})"
        )


def quantile_backtest(
    close: Any,
    factor: Any,
    quantiles: int = 5,
    rebalance: int = 5,
    lag: int = 1,
    long_short: bool = True,
    fees: float = 0.0,
    slippage: float = 0.0,
    init_cash: float = 1e6,
    **kwargs: Any,
) -> QuantileBacktestResult:
    """Simulate holding each quantile of ``factor``, and optionally a long-short book.

    Labels come from :func:`quantile_labels`. Every ``rebalance`` bars,
    starting at bar 0, each group ``g`` targets an equal weight of the names
    in that group and zero elsewhere. With ``long_short``, one more group
    targets ``+0.5 / count`` on the top group and ``-0.5 / count`` on the
    bottom group. Other bars are NaN, so the positions hold. Targets are then
    shifted forward by ``lag`` bars.

    The groups share no cash with each other. Each one is its own account
    over the same symbols (``cash_sharing`` inside the group, ``column_map``
    pointing every group at ``close``). Orders are target weights, sold
    before bought. ``**kwargs`` are forwarded to :func:`phasor.from_orders`;
    the sizing arguments this function sets take precedence over them.
    """
    from . import from_orders

    close_arr = _panel(close, "close")
    factor_arr = _panel(factor, "factor")
    if factor_arr.shape != close_arr.shape:
        raise ValueError(f"factor has shape {factor_arr.shape}, expected {close_arr.shape}")
    n_bars, n_symbols = close_arr.shape
    groups = _at_least(quantiles, 1, "quantiles")
    every = _at_least(rebalance, 1, "rebalance")
    delay = _count(lag, "lag")
    long_short = bool(long_short)
    cash = float(init_cash)

    labels = quantile_labels(factor_arr, groups)
    active = np.zeros(n_bars, dtype=bool)
    if n_bars:
        active[::every] = True
    pieces = [_group_targets(labels == group, 1.0, active) for group in range(groups)]
    num_groups = groups
    if long_short:
        pieces.append(
            _group_targets(labels == (groups - 1), 0.5, active)
            + _group_targets(labels == 0, -0.5, active)
        )
        num_groups += 1
    raw = np.asfortranarray(np.concatenate(pieces, axis=1))
    if delay == 0:
        shifted = raw
    else:
        shifted = np.full(raw.shape, np.nan, dtype=np.float64, order="F")
        if delay < n_bars:
            shifted[delay:] = raw[: n_bars - delay]

    options = {key: value for key, value in kwargs.items() if key != "size"}
    options.update(
        size_type="targetpercent",
        direction="both" if long_short else "longonly",
        cash_sharing=True,
        group_lens=[n_symbols] * num_groups,
        column_map=np.tile(np.arange(n_symbols, dtype=np.int64), num_groups),
        init_cash=[cash] * num_groups,
        fees=float(fees),
        slippage=float(slippage),
        call_seq="auto",
    )
    portfolio = from_orders(close_arr, shifted, **options)
    return QuantileBacktestResult(portfolio, num_groups, groups, long_short)
