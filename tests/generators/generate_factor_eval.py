#!/usr/bin/env python3
"""Factor-evaluation statistics: an independent reference for the golden fixture.

Pins the operators behind ``tests/fixtures/factor_eval.tsv``. The
implementation is pandas, SciPy, statsmodels and NumPy only. It shares no
code with phasor and does not import the Rust extension.

Each row of the fixture is one case. ``x`` and ``y`` are matrices flattened
in column-major (Fortran) order; ``y`` is ``-`` when the operator has no
second input. Floats are ``repr(float(v))`` (shortest round-trip). NaN is
``nan`` and infinities are ``inf`` / ``-inf``.

What each operator pins:

* ``forward_returns`` — ``close[t+lag+h] / close[t+lag] - 1``, as
  ``DataFrame.shift(-(lag+h)) / DataFrame.shift(-lag) - 1``. A non-finite
  close is missing. The result is missing once ``t+lag+h`` falls off the
  panel.
* ``cs_pearson`` / ``cs_spearman`` — one correlation per row, on the columns
  where both inputs are finite. Fewer than ``min`` pairs is missing. A side
  with zero sample variance is missing; for Spearman that test is on the
  average ranks of the kept pairs (``scipy.stats.spearmanr``).
* ``ts_pearson`` / ``ts_spearman`` — the same calculation down each column.
* ``rolling_pearson`` / ``rolling_spearman`` — for each column, the window
  at row ``t`` is ``max(0, t-w+1) .. t`` inclusive. Finite pairs inside that
  window are correlated with SciPy. Spearman ranks are recomputed on those
  pairs. Implemented as an explicit window loop.
* ``summary`` — drop NaN, then eight numbers: mean, sample standard
  deviation (ddof=1), ``mean/std``, ordinary ``t = icir*sqrt(n)``, the
  Newey-West ``t`` of the mean, its two-sided normal p-value, the fraction
  of values strictly above 0, and ``n``. The Newey-West ``t`` is
  ``statsmodels`` ``OLS`` with ``cov_type="HAC"``, Bartlett kernel,
  ``use_correction=False``. The script also computes

  ``gamma_l = (1/n) * sum_{t=l}^{n-1} (x_t-mean)(x_{t-l}-mean)``,
  ``omega = gamma_0 + 2 * sum_{l=1}^{lags} (1 - l/(lags+1)) * gamma_l``,
  ``t = mean / sqrt(omega/n)``

  and requires the two ``t`` values to agree to ``1e-10`` relative.
  ``n == 0``, ``n == 1``, a zero sample standard deviation and ``omega <= 0``
  are filled in by hand and do not call statsmodels.

Panels are small and awkward on purpose: missing rows and columns, too few
finite names, ties, constants, huge and tiny magnitudes, one and two
columns, perfect correlation and its opposite. Draws come from
``np.random.default_rng(20261029)``.

Usage::

    uv run --no-project --with numpy --with pandas --with scipy --with statsmodels \\
        python tests/generators/generate_factor_eval.py
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import statsmodels.api as sm

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "factor_eval.tsv"
SEED = 20261029
HEADER = ("case", "op", "params", "n_rows", "n_cols", "x", "y", "expected")

_ROW_OPS = {"cs_pearson", "cs_spearman"}
_COL_OPS = {"ts_pearson", "ts_spearman"}
_MATRIX_OPS = {"forward_returns", "rolling_pearson", "rolling_spearman"}


def fmt_float(v: float) -> str:
    return repr(float(v))


def fmt_vec(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float64).ravel(order="F")
    return ",".join(fmt_float(v) for v in flat)


def fmt_params(pairs: list[tuple[str, int]] | None) -> str:
    if not pairs:
        return "-"
    return ",".join(f"{key}={int(value)}" for key, value in pairs)


def add(
    cases: list[tuple[str, ...]],
    case: str,
    op: str,
    params: list[tuple[str, int]] | None,
    x: np.ndarray,
    y: np.ndarray | None,
    expected: np.ndarray,
) -> np.ndarray:
    """Append one fixture row. ``x`` is stored column-major; ``y`` may be absent."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"{case}: x must be a matrix, got shape {x.shape}")
    n_rows, n_cols = x.shape
    if y is None:
        y_text = "-"
    else:
        y = np.asarray(y, dtype=np.float64)
        if y.shape != x.shape:
            raise ValueError(f"{case}: x {x.shape} and y {y.shape} differ")
        y_text = fmt_vec(y)
    expected = np.asarray(expected, dtype=np.float64).ravel(order="F")
    if op == "summary":
        if expected.size != 8 or n_cols != 1:
            raise ValueError(f"{case}: summary shape {x.shape} expected {expected.size}")
    elif op in _ROW_OPS and expected.size != n_rows:
        raise ValueError(f"{case}: expected {expected.size} values, wanted {n_rows} rows")
    elif op in _COL_OPS and expected.size != n_cols:
        raise ValueError(f"{case}: expected {expected.size} values, wanted {n_cols} cols")
    elif op in _MATRIX_OPS and expected.size != n_rows * n_cols:
        raise ValueError(f"{case}: expected length {expected.size}, panel {n_rows}x{n_cols}")
    elif op not in _ROW_OPS | _COL_OPS | _MATRIX_OPS | {"summary"}:
        raise ValueError(f"{case}: unknown op {op}")
    row = (
        case,
        op,
        fmt_params(params),
        str(n_rows),
        str(n_cols),
        fmt_vec(x),
        y_text,
        fmt_vec(expected),
    )
    if any("\t" in field or "\n" in field or "\r" in field for field in row):
        raise ValueError(f"{case}: field contains a separator")
    cases.append(row)
    return expected


def forward_returns(close: np.ndarray, h: int, lag: int) -> np.ndarray:
    """``shift(-(lag+h)) / shift(-lag) - 1``, after non-finite closes become NaN."""
    values = np.array(close, dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    frame = pd.DataFrame(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = frame.shift(-(int(lag) + int(h))) / frame.shift(-int(lag)) - 1.0
    return out.to_numpy(dtype=np.float64)


def _forward_reference(close: np.ndarray, h: int, lag: int) -> np.ndarray:
    """The same ratio as an index loop, used only to check the pandas path."""
    values = np.array(close, dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    n_rows, n_cols = values.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for t in range(n_rows):
        num = t + lag + h
        den = t + lag
        if num < n_rows:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[t, :] = values[num, :] / values[den, :] - 1.0
    return out


def _zero_var(values: np.ndarray) -> bool:
    """True when the sample variance is zero or undefined (fewer than 2 points)."""
    if values.size < 2:
        return True
    var = float(np.var(values, ddof=1))
    return (not math.isfinite(var)) or var == 0.0


def _correlate(left: np.ndarray, right: np.ndarray, min_obs: int, method: str) -> float:
    """Pearson or Spearman on the jointly finite entries of one slice."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if left.size < min_obs:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "spearman":
            # Average ranks, recomputed on this kept set only.
            rank_left = scipy.stats.rankdata(left, method="average")
            rank_right = scipy.stats.rankdata(right, method="average")
            if _zero_var(rank_left) or _zero_var(rank_right):
                return math.nan
            score = float(scipy.stats.spearmanr(left, right)[0])
        elif method == "pearson":
            if _zero_var(left) or _zero_var(right):
                return math.nan
            score = float(scipy.stats.pearsonr(left, right)[0])
        else:
            raise ValueError(method)
    if not math.isfinite(score):
        return math.nan
    return score


def cs_corr(x: np.ndarray, y: np.ndarray, min_obs: int, method: str) -> np.ndarray:
    n_rows = x.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    for t in range(n_rows):
        out[t] = _correlate(x[t, :], y[t, :], min_obs, method)
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, min_obs: int, method: str) -> np.ndarray:
    n_cols = x.shape[1]
    out = np.empty(n_cols, dtype=np.float64)
    for c in range(n_cols):
        out[c] = _correlate(x[:, c], y[:, c], min_obs, method)
    return out


def rolling_corr(
    x: np.ndarray, y: np.ndarray, window: int, min_obs: int, method: str
) -> np.ndarray:
    """Trailing window ``max(0, t-w+1) ..= t``, one SciPy call per window."""
    n_rows, n_cols = x.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    for c in range(n_cols):
        for t in range(n_rows):
            start = t - window + 1
            if start < 0:
                start = 0
            out[t, c] = _correlate(x[start : t + 1, c], y[start : t + 1, c], min_obs, method)
    return out


def _gamma(demeaned: np.ndarray, lag: int) -> float:
    n = demeaned.size
    if lag >= n:
        return 0.0
    return float(np.dot(demeaned[lag:], demeaned[: n - lag]) / n)


def summary_stats(series: np.ndarray, lags: int) -> np.ndarray:
    """Eight summary numbers. Degenerate samples never reach statsmodels."""
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    values = values[~np.isnan(values)]
    n = int(values.size)
    nan = math.nan
    if n == 0:
        return np.array([nan, nan, nan, nan, nan, nan, nan, 0.0], dtype=np.float64)
    mean = float(np.mean(values))
    hit = float(np.mean(values > 0.0))
    if n == 1:
        return np.array([mean, nan, nan, nan, nan, nan, hit, 1.0], dtype=np.float64)
    std = float(np.std(values, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        icir = nan
        t_stat = nan
    else:
        icir = mean / std
        t_stat = icir * math.sqrt(n)
    demeaned = values - mean
    omega = _gamma(demeaned, 0)
    for lag in range(1, int(lags) + 1):
        weight = 1.0 - lag / (int(lags) + 1.0)
        omega += 2.0 * weight * _gamma(demeaned, lag)
    if not omega > 0.0:
        t_nw = nan
        p_nw = nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.OLS(values, np.ones(n)).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": int(lags), "use_correction": False},
            )
        t_sm = float(fitted.params[0] / fitted.bse[0])
        t_direct = mean / math.sqrt(omega / n)
        scale = max(abs(t_sm), abs(t_direct))
        if scale == 0.0:
            agree = t_sm == t_direct
        else:
            agree = abs(t_sm - t_direct) / scale <= 1e-10
        if not agree:
            raise AssertionError(
                f"Newey-West t disagrees: statsmodels={t_sm!r} direct={t_direct!r} lags={lags}"
            )
        t_nw = t_sm
        p_nw = float(scipy.special.erfc(abs(t_nw) / math.sqrt(2.0)))
    return np.array([mean, std, icir, t_stat, t_nw, p_nw, hit, float(n)], dtype=np.float64)


def _panel(rng: np.random.Generator, rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, scale, size=(rows, cols))


def _prices(rng: np.random.Generator, rows: int, cols: int, vol: float = 0.02) -> np.ndarray:
    shocks = rng.normal(0.0, vol, size=(rows, cols))
    return np.exp(np.cumsum(shocks, axis=0)) * 50.0


def _punch(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, n_x: int, n_y: int
) -> tuple[np.ndarray, np.ndarray]:
    """Write NaN into ``n_x`` cells of ``x`` and ``n_y`` different cells of ``y``."""
    x = np.array(x, dtype=np.float64, copy=True)
    y = np.array(y, dtype=np.float64, copy=True)
    spots = rng.choice(x.size, size=n_x + n_y, replace=False)
    x.ravel()[spots[:n_x]] = np.nan
    y.ravel()[spots[n_x:]] = np.nan
    return x, y


def _add_corr(
    cases: list[tuple[str, ...]],
    scope: str,
    stem: str,
    x: np.ndarray,
    y: np.ndarray,
    min_obs: int,
    window: int | None = None,
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    outs: dict[str, np.ndarray] = {}
    for method in ("pearson", "spearman"):
        if scope == "cs":
            op = f"cs_{method}"
            case = f"cs_{method}_{stem}"
            params: list[tuple[str, int]] = [("min", min_obs)]
            expected = cs_corr(x, y, min_obs, method)
        elif scope == "ts":
            op = f"ts_{method}"
            case = f"ts_{method}_{stem}"
            params = [("min", min_obs)]
            expected = ts_corr(x, y, min_obs, method)
        elif scope == "rolling":
            if window is None:
                raise ValueError(stem)
            op = f"rolling_{method}"
            case = f"rolling_{method}_{stem}"
            params = [("w", window), ("min", min_obs)]
            expected = rolling_corr(x, y, window, min_obs, method)
        else:
            raise ValueError(scope)
        add(cases, case, op, params, x, y, expected)
        outs[method] = expected
    return outs


def _assert_close(actual: np.ndarray, want: float, case: str, where: np.ndarray | None = None) -> None:
    values = np.asarray(actual, dtype=np.float64).ravel()
    if where is None:
        picked = values
    else:
        picked = values[np.asarray(where).ravel()]
    if picked.size == 0 or not np.all(np.isfinite(picked)):
        raise AssertionError(f"{case}: missing a finite result")
    if not np.all(np.abs(picked - want) <= 1e-8):
        raise AssertionError(f"{case}: expected {want}, got {picked}")


def _assert_nan(actual: np.ndarray, case: str, where: np.ndarray | None = None) -> None:
    values = np.asarray(actual, dtype=np.float64).ravel()
    if where is None:
        picked = values
    else:
        picked = values[np.asarray(where).ravel()]
    if picked.size == 0 or np.any(np.isfinite(picked)):
        raise AssertionError(f"{case}: expected NaN, got {picked}")


def _assert_differ(left: np.ndarray, right: np.ndarray, case: str) -> None:
    both = np.isfinite(left) & np.isfinite(right)
    if not np.any(both) or np.allclose(left[both], right[both], rtol=0.0, atol=1e-6):
        raise AssertionError(f"{case}: pearson and spearman did not separate")


def _forward_cases(cases: list[tuple[str, ...]], rng: np.random.Generator) -> None:
    close = _prices(rng, 12, 4)
    close[:, 3] = 80.0  # flat prices: a defined return of 0, then a missing tail
    out = forward_returns(close, 1, 0)
    add(cases, "fr_h1_lag0", "forward_returns", [("h", 1), ("lag", 0)], close, None, out)
    if not np.allclose(out[:-1, 3], 0.0) or not math.isnan(out[-1, 3]):
        raise AssertionError("fr_h1_lag0: flat column")

    close = _prices(rng, 15, 3, vol=0.03)
    out = forward_returns(close, 3, 1)
    add(cases, "fr_h3_lag1", "forward_returns", [("h", 3), ("lag", 1)], close, None, out)
    if np.any(np.isfinite(out[-4:, :])):
        raise AssertionError("fr_h3_lag1: tail should be off the panel")

    close = _prices(rng, 6, 2)
    out = forward_returns(close, 20, 1)
    add(cases, "fr_h_exceeds", "forward_returns", [("h", 20), ("lag", 1)], close, None, out)
    _assert_nan(out, "fr_h_exceeds")

    close = _prices(rng, 10, 4)
    close[1, 0] = np.nan
    close[3, 1] = np.nan
    close[6, 2] = np.nan
    close[2, :] = np.nan  # an entire bar missing
    close[:, 3] = np.nan  # an entire name missing
    close[7, 0] = np.inf
    close[8, 1] = -np.inf
    out = forward_returns(close, 1, 0)
    add(cases, "fr_nan_closes", "forward_returns", [("h", 1), ("lag", 0)], close, None, out)
    _assert_nan(out[:, 3], "fr_nan_closes column")

    close = _prices(rng, 9, 1)
    add(
        cases,
        "fr_ncols1",
        "forward_returns",
        [("h", 1), ("lag", 0)],
        close,
        None,
        forward_returns(close, 1, 0),
    )
    close = _prices(rng, 8, 2, vol=0.04)
    add(
        cases,
        "fr_ncols2",
        "forward_returns",
        [("h", 2), ("lag", 1)],
        close,
        None,
        forward_returns(close, 2, 1),
    )

    close = _prices(rng, 8, 2, vol=0.01)
    close[:, 0] *= 1e12
    close[:, 1] *= 1e-12
    add(
        cases,
        "fr_huge_tiny",
        "forward_returns",
        [("h", 1), ("lag", 0)],
        close,
        None,
        forward_returns(close, 1, 0),
    )

    # Zero and negative closes: +inf, -inf, 0/0, and a return of exactly -1.
    close = np.array(
        [
            [10.0, 8.0],
            [12.0, 0.0],
            [0.0, 0.0],
            [5.0, -4.0],
            [5.0, 2.0],
            [7.0, 2.0],
        ],
        dtype=np.float64,
    )
    out = forward_returns(close, 1, 0)
    add(cases, "fr_zero_close", "forward_returns", [("h", 1), ("lag", 0)], close, None, out)
    if out[0, 1] != -1.0 or out[1, 0] != -1.0 or out[3, 0] != 0.0 or out[3, 1] != -1.5:
        raise AssertionError(f"fr_zero_close values {out}")
    if not (math.isinf(out[2, 0]) and out[2, 0] > 0 and math.isinf(out[2, 1]) and out[2, 1] < 0):
        raise AssertionError(f"fr_zero_close infinities {out[2]}")
    if not math.isnan(out[1, 1]):
        raise AssertionError("fr_zero_close 0/0")


def _cs_cases(cases: list[tuple[str, ...]], rng: np.random.Generator) -> None:
    x = _panel(rng, 12, 5)
    y = _panel(rng, 12, 5)
    outs = _add_corr(cases, "cs", "plain", x, y, min_obs=4)
    if not np.all(np.isfinite(outs["pearson"])):
        raise AssertionError("cs plain")

    x, y = _punch(rng, _panel(rng, 10, 5), _panel(rng, 10, 5), 8, 7)
    x[0, 1] = np.inf
    y[2, 3] = -np.inf
    _add_corr(cases, "cs", "nans", x, y, min_obs=3)

    x = _panel(rng, 8, 4)
    y = _panel(rng, 8, 4)
    x[2, :] = np.nan
    y[5, :] = np.nan
    x[:, 3] = np.nan  # one name missing on every row; min=3 still leaves 3 names
    outs = _add_corr(cases, "cs", "allnan_row", x, y, min_obs=3)
    _assert_nan(outs["pearson"], "cs allnan", where=np.array([2, 5]))
    _assert_close_finite = np.isfinite(outs["pearson"][[0, 1, 3, 4, 6, 7]]).all()
    if not _assert_close_finite:
        raise AssertionError("cs allnan_row: a full row collapsed")

    x = _panel(rng, 7, 6)
    y = _panel(rng, 7, 6)
    x[1, 0] = np.nan
    y[1, 1] = np.nan  # 4 finite pairs < min 5
    x[2, 1:] = np.nan  # 1 finite pair
    y[3, :] = np.nan
    x[4, 5] = np.nan  # exactly 5 pairs, should still compute
    outs = _add_corr(cases, "cs", "below_min", x, y, min_obs=5)
    _assert_nan(outs["spearman"], "cs below_min", where=np.array([1, 2, 3]))
    if not np.isfinite(outs["pearson"][0]) or not np.isfinite(outs["pearson"][4]):
        raise AssertionError("cs below_min: a long enough row is missing")

    # Constants, including one that is constant only on the jointly finite names.
    x = _panel(rng, 6, 5)
    y = _panel(rng, 6, 5)
    x[0, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y[0, :] = 3.0 * x[0, :] + 1.0
    x[1, :] = 1.0e12
    y[2, :] = -4.0
    x[3, :] = 1.0e-12
    x[3, 4] = 42.0
    y[3, 4] = np.nan  # kept x is the constant 1e-12
    x[4, :] = 7.0
    y[4, :] = 7.0
    outs = _add_corr(cases, "cs", "constant", x, y, min_obs=3)
    _assert_close(outs["pearson"][0], 1.0, "cs constant anchor")
    _assert_nan(outs["pearson"], "cs constant", where=np.array([1, 2, 3, 4]))
    _assert_nan(outs["spearman"], "cs constant ranks", where=np.array([1, 2, 3, 4]))

    base_x = _panel(rng, 10, 4)
    base_y = _panel(rng, 10, 4)
    x = base_x * 1.0e12
    y = base_y * 1.0e-12
    x[3, :] = 1.0e12
    y[4, :] = 1.0e-12
    outs = _add_corr(cases, "cs", "huge_tiny", x, y, min_obs=3)
    _assert_nan(outs["pearson"], "cs huge constant", where=np.array([3, 4]))
    if not np.all(np.isfinite(outs["spearman"][[0, 1, 2, 5]])):
        raise AssertionError("cs huge_tiny: scaled rows did not correlate")

    x = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [5.0, 5.0],
            [1.0, 4.0],
            [1.0e12, 1.0e12 + 5.0],
            [1.0e-12, 3.0e-12],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [3.0, 7.0],
            [7.0, 3.0],
            [1.0, 4.0],
            [np.nan, 2.0],
            [1.0, 2.0],
            [-1.0, -4.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    outs = _add_corr(cases, "cs", "ncols2", x, y, min_obs=2)
    _assert_close(outs["pearson"], 1.0, "cs ncols2 +", where=np.array([0, 4]))
    _assert_close(outs["pearson"], -1.0, "cs ncols2 -", where=np.array([1, 5]))
    _assert_close(outs["spearman"], 1.0, "cs ncols2 rank +", where=np.array([0, 4]))
    _assert_nan(outs["pearson"], "cs ncols2 nan", where=np.array([2, 3, 6]))

    x = _panel(rng, 6, 1)
    y = 3.0 * x + 1.0
    x[2, 0] = np.nan
    y[4, 0] = np.inf
    outs = _add_corr(cases, "cs", "ncols1", x, y, min_obs=1)
    _assert_nan(outs["pearson"], "cs ncols1")
    _assert_nan(outs["spearman"], "cs ncols1 rank")

    x = _panel(rng, 8, 5)
    y = 3.0 * x + 1.0
    x[0, 1] = np.nan
    y[0, 3] = np.nan
    y[1, 0] = np.nan
    x[1, 2] = np.nan
    x[2, :] = np.nan
    y[3, :] = np.nan
    x[4, :3] = np.nan
    y[4, 2:] = np.nan  # no column where both survive
    x[7, 4] = np.inf
    y[6, 0] = -np.inf
    outs = _add_corr(cases, "cs", "perfect", x, y, min_obs=3)
    _assert_close(outs["pearson"], 1.0, "cs perfect", where=np.array([0, 1, 5, 6, 7]))
    _assert_close(outs["spearman"], 1.0, "cs perfect rank", where=np.array([0, 1, 5, 6, 7]))
    _assert_nan(outs["pearson"], "cs perfect nan", where=np.array([2, 3, 4]))

    x = _panel(rng, 8, 5)
    y = -3.0 * x + 1.0
    x[0, 2] = np.nan
    y[1, 4] = np.nan
    y[2, :] = np.nan
    x[5, 0] = np.nan
    y[5, 1] = np.inf
    outs = _add_corr(cases, "cs", "anti", x, y, min_obs=3)
    _assert_close(outs["pearson"], -1.0, "cs anti", where=np.array([0, 1, 3, 4, 5, 6, 7]))
    _assert_close(outs["spearman"], -1.0, "cs anti rank", where=np.array([0, 1, 3, 4, 6, 7]))
    _assert_nan(outs["pearson"], "cs anti nan", where=np.array([2]))

    # One-decimal ties. Row 1 is the average-rank case: y moves inside a tie.
    x = np.array(
        [
            [0.1, 0.1, 0.2, 0.3, 0.4, 0.4],
            [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.2, 0.2, 0.3, 0.3, 0.4, np.nan],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.9],
            [0.1, 0.1, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [0.1, 0.9, 0.2, 0.3, 0.4, 0.8],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 1.0, 2.0, 2.0, 3.0, 9.0],
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            [0.1, 5.0, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ],
        dtype=np.float64,
    )
    outs = _add_corr(cases, "cs", "ties", x, y, min_obs=4)
    _assert_differ(outs["pearson"], outs["spearman"], "cs ties")
    _assert_nan(outs["spearman"], "cs ties nan", where=np.array([2, 6]))
    _assert_close(outs["spearman"][4], 1.0, "cs ties monotone")
    _assert_close(outs["pearson"][4], 1.0, "cs ties monotone pearson")


def _ts_cases(cases: list[tuple[str, ...]], rng: np.random.Generator) -> None:
    x = _panel(rng, 16, 4)
    y = _panel(rng, 16, 4)
    outs = _add_corr(cases, "ts", "plain", x, y, min_obs=8)
    if not np.all(np.isfinite(outs["pearson"])):
        raise AssertionError("ts plain")

    x, y = _punch(rng, _panel(rng, 14, 4), _panel(rng, 14, 4), 8, 6)
    x[3, 0] = np.inf
    y[5, 1] = -np.inf
    _add_corr(cases, "ts", "nans", x, y, min_obs=6)

    x = _panel(rng, 12, 4)
    y = _panel(rng, 12, 4)
    x[:, 1] = np.nan
    y[:, 3] = np.nan
    outs = _add_corr(cases, "ts", "allnan_col", x, y, min_obs=6)
    _assert_nan(outs["pearson"], "ts allnan", where=np.array([1, 3]))
    if not np.all(np.isfinite(outs["spearman"][[0, 2]])):
        raise AssertionError("ts allnan_col: a live column collapsed")

    x = _panel(rng, 12, 3)
    y = _panel(rng, 12, 3)
    x[:7, 1] = np.nan  # 5 finite pairs < min 8
    y[::2, 2] = np.nan  # rows 0,2,...,10 missing -> 6 finite pairs
    outs = _add_corr(cases, "ts", "below_min", x, y, min_obs=8)
    _assert_nan(outs["pearson"], "ts below_min", where=np.array([1, 2]))
    if not np.isfinite(outs["pearson"][0]):
        raise AssertionError("ts below_min: full column")

    x = _panel(rng, 12, 4)
    y = _panel(rng, 12, 4)
    x[:, 1] = 5.0
    y[:, 2] = -2.0
    x[:, 3] = 1.0e12
    x[2, 3] = -7.0
    x[9, 3] = 4.0
    y[2, 3] = np.nan
    y[9, 3] = np.nan  # kept x is the constant 1e12
    outs = _add_corr(cases, "ts", "constant", x, y, min_obs=6)
    if not np.isfinite(outs["pearson"][0]):
        raise AssertionError("ts constant: live column")
    _assert_nan(outs["pearson"], "ts constant", where=np.array([1, 2, 3]))
    _assert_nan(outs["spearman"], "ts constant ranks", where=np.array([1, 2, 3]))

    base_x = _panel(rng, 12, 3)
    base_y = _panel(rng, 12, 3)
    x = base_x * 1.0e12
    y = base_y * 1.0e-12
    x[:, 2] = 1.0e-12
    outs = _add_corr(cases, "ts", "huge_tiny", x, y, min_obs=6)
    _assert_nan(outs["spearman"], "ts huge constant", where=np.array([2]))
    if not np.all(np.isfinite(outs["pearson"][:2])):
        raise AssertionError("ts huge_tiny")

    x = _panel(rng, 10, 2)
    y = np.column_stack((3.0 * x[:, 0] + 1.0, -3.0 * x[:, 1] + 1.0))
    x[0, 0] = np.nan
    y[1, 0] = np.nan
    y[2, 1] = np.nan
    x[3, 1] = np.inf
    outs = _add_corr(cases, "ts", "ncols2", x, y, min_obs=4)
    _assert_close(outs["pearson"][0], 1.0, "ts ncols2 +")
    _assert_close(outs["pearson"][1], -1.0, "ts ncols2 -")
    _assert_close(outs["spearman"][0], 1.0, "ts ncols2 rank +")
    _assert_close(outs["spearman"][1], -1.0, "ts ncols2 rank -")

    x = _panel(rng, 12, 1)
    y = 0.5 * x + _panel(rng, 12, 1, scale=0.2)
    x[4, 0] = np.nan
    y[8, 0] = np.nan
    outs = _add_corr(cases, "ts", "ncols1", x, y, min_obs=6)
    if not np.isfinite(outs["pearson"][0]) or not np.isfinite(outs["spearman"][0]):
        raise AssertionError("ts ncols1")

    x = _panel(rng, 14, 4)
    y = 3.0 * x + 1.0
    x[1, 0] = np.nan
    y[2, 0] = np.nan
    y[4, 1] = np.inf
    x[0, 3] = -np.inf
    outs = _add_corr(cases, "ts", "perfect", x, y, min_obs=6)
    _assert_close(outs["pearson"], 1.0, "ts perfect")
    _assert_close(outs["spearman"], 1.0, "ts perfect rank")

    x = _panel(rng, 14, 4)
    y = -3.0 * x + 1.0
    x[3, 1] = np.nan
    y[6, 2] = np.nan
    y[:, 3] = np.nan
    outs = _add_corr(cases, "ts", "anti", x, y, min_obs=6)
    _assert_close(outs["pearson"], -1.0, "ts anti", where=np.array([0, 1, 2]))
    _assert_close(outs["spearman"], -1.0, "ts anti rank", where=np.array([0, 1, 2]))
    _assert_nan(outs["pearson"], "ts anti col", where=np.array([3]))

    x = np.array(
        [
            [0.1, 0.2, 0.5],
            [0.1, 0.2, 0.5],
            [0.1, 0.3, 0.5],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
            [0.2, 0.4, 0.5],
            [0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5],
            [0.3, 0.5, 0.5],
            [0.4, 0.5, 0.5],
            [0.4, 0.1, 0.5],
            [0.5, 0.1, 0.5],
            [0.5, 0.1, 0.5],
            [0.5, 0.2, 0.5],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [0.2, 1.0, 1.0],
            [0.8, 1.0, 2.0],
            [0.1, 2.0, 3.0],
            [0.4, 3.0, 4.0],
            [0.9, 2.0, 5.0],
            [0.3, 4.0, 6.0],
            [0.6, 4.0, 7.0],
            [1.1, 5.0, 8.0],
            [0.2, 3.0, 9.0],
            [0.7, 6.0, 1.0],
            [1.0, 0.5, 2.0],
            [0.4, 0.2, 3.0],
            [1.3, 0.4, np.nan],
            [0.5, 1.5, 4.0],
        ],
        dtype=np.float64,
    )
    outs = _add_corr(cases, "ts", "ties", x, y, min_obs=8)
    _assert_differ(outs["pearson"], outs["spearman"], "ts ties")
    _assert_nan(outs["spearman"], "ts ties constant", where=np.array([2]))
    if not np.isfinite(outs["pearson"][0]) or not np.isfinite(outs["spearman"][1]):
        raise AssertionError("ts ties: a tied column collapsed")


def _rolling_cases(cases: list[tuple[str, ...]], rng: np.random.Generator) -> None:
    x = _panel(rng, 12, 3)
    y = _panel(rng, 12, 3)
    outs = _add_corr(cases, "rolling", "plain", x, y, min_obs=3, window=5)
    _assert_nan(outs["pearson"][:2, :], "rolling plain warmup")
    if not np.all(np.isfinite(outs["spearman"][4:, :])):
        raise AssertionError("rolling plain: a full window collapsed")

    x, y = _punch(rng, _panel(rng, 12, 3), _panel(rng, 12, 3), 7, 6)
    x[4, 0] = np.inf
    y[6, 1] = -np.inf
    _add_corr(cases, "rolling", "nans", x, y, min_obs=3, window=5)

    x = _panel(rng, 10, 3)
    y = _panel(rng, 10, 3)
    x[:, 1] = np.nan
    y[:, 2] = np.nan
    outs = _add_corr(cases, "rolling", "allnan_col", x, y, min_obs=3, window=4)
    _assert_nan(outs["pearson"][:, 1], "rolling allnan x")
    _assert_nan(outs["spearman"][:, 2], "rolling allnan y")
    if not np.any(np.isfinite(outs["pearson"][:, 0])):
        raise AssertionError("rolling allnan_col: live column")

    # Window length 5, min 5. Column 1 is too sparse to ever qualify.
    x = _panel(rng, 10, 2)
    y = _panel(rng, 10, 2)
    x[::2, 1] = np.nan
    y[1::2, 1] = np.nan
    outs = _add_corr(cases, "rolling", "below_min", x, y, min_obs=5, window=5)
    _assert_nan(outs["pearson"][:4, 0], "rolling below warmup")
    _assert_nan(outs["spearman"][:, 1], "rolling below sparse")
    if not np.all(np.isfinite(outs["pearson"][4:, 0])):
        raise AssertionError("rolling below_min: dense window")

    x = _panel(rng, 10, 3)
    y = _panel(rng, 10, 3)
    x[:, 1] = 3.0
    y[:, 2] = 1.0e12
    # Column 0 stays random and should produce finite windows once it is long enough.
    outs = _add_corr(cases, "rolling", "constant", x, y, min_obs=3, window=5)
    _assert_nan(outs["pearson"][:, 1], "rolling constant x")
    _assert_nan(outs["spearman"][:, 2], "rolling constant y")
    if not np.any(np.isfinite(outs["pearson"][:, 0])):
        raise AssertionError("rolling constant: live column")

    x = _panel(rng, 10, 2) * 1.0e12
    y = _panel(rng, 10, 2) * 1.0e-12
    outs = _add_corr(cases, "rolling", "huge_tiny", x, y, min_obs=3, window=4)
    _assert_nan(outs["pearson"][:2, :], "rolling huge warmup")
    if not np.all(np.isfinite(outs["spearman"][5:, :])):
        raise AssertionError("rolling huge_tiny")

    x = _panel(rng, 9, 2)
    y = np.column_stack((3.0 * x[:, 0] + 1.0, -3.0 * x[:, 1] + 1.0))
    x[2, 0] = np.nan
    y[4, 1] = np.nan
    outs = _add_corr(cases, "rolling", "ncols2", x, y, min_obs=3, window=4)
    # Column 0 is exactly linear except one hole, so late windows are +1.
    _assert_close(outs["pearson"][-1, 0], 1.0, "rolling ncols2 +")
    _assert_close(outs["spearman"][-1, 0], 1.0, "rolling ncols2 rank +")
    _assert_close(outs["pearson"][3, 1], -1.0, "rolling ncols2 -")

    x = _panel(rng, 11, 1)
    y = 0.4 * x + _panel(rng, 11, 1, scale=0.15)
    x[5, 0] = np.nan
    outs = _add_corr(cases, "rolling", "ncols1", x, y, min_obs=3, window=5)
    _assert_nan(outs["pearson"][:2, 0], "rolling ncols1 warmup")
    if not np.isfinite(outs["spearman"][-1, 0]):
        raise AssertionError("rolling ncols1")

    x = _panel(rng, 10, 3)
    y = 3.0 * x + 1.0
    x[3, 0] = np.nan
    y[4, 1] = np.inf
    outs = _add_corr(cases, "rolling", "perfect", x, y, min_obs=3, window=4)
    _assert_close(outs["pearson"][-1, :], 1.0, "rolling perfect tail")
    _assert_close(outs["spearman"][-1, 2], 1.0, "rolling perfect rank")
    _assert_nan(outs["pearson"][:2, 2], "rolling perfect warmup")

    x = _panel(rng, 10, 3)
    y = -3.0 * x + 1.0
    y[2, 0] = np.nan
    x[6, 1] = -np.inf
    outs = _add_corr(cases, "rolling", "anti", x, y, min_obs=3, window=4)
    _assert_close(outs["pearson"][-1, :], -1.0, "rolling anti tail")
    _assert_close(outs["spearman"][-1, 2], -1.0, "rolling anti rank")

    x = np.array(
        [
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 2.0],
            [0.2, 2.0],
            [0.2, 2.0],
            [0.3, 3.0],
            [0.3, 3.0],
            [0.3, 4.0],
            [0.4, 4.0],
            [0.4, 5.0],
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            [0.2, 1.0],
            [0.9, 4.0],
            [0.1, 2.0],
            [0.4, 2.5],
            [0.8, 5.0],
            [0.3, 3.0],
            [0.7, 6.0],
            [0.2, 4.0],
            [1.0, 7.0],
            [0.5, 5.0],
        ],
        dtype=np.float64,
    )
    outs = _add_corr(cases, "rolling", "ties", x, y, min_obs=4, window=6)
    _assert_differ(outs["pearson"], outs["spearman"], "rolling ties")
    _assert_nan(outs["pearson"][:3, :], "rolling ties warmup")

    x = _panel(rng, 6, 3)
    y = _panel(rng, 6, 3)
    outs = _add_corr(cases, "rolling", "w1", x, y, min_obs=2, window=1)
    _assert_nan(outs["pearson"], "rolling w1")
    _assert_nan(outs["spearman"], "rolling w1 rank")

    # w past the sample is an expanding window. min=2, so the first row is short.
    x = _panel(rng, 6, 3)
    y = np.column_stack((3.0 * x[:, 0] + 1.0, -3.0 * x[:, 1] + 1.0, 3.0 * x[:, 2] + 1.0))
    x[0, 2] = np.nan
    y[1, 2] = np.nan
    outs = _add_corr(cases, "rolling", "w_exceeds", x, y, min_obs=2, window=30)
    _assert_nan(outs["pearson"][0, :], "rolling w_exceeds first")
    _assert_close(outs["pearson"][1:, 0], 1.0, "rolling w_exceeds +")
    _assert_close(outs["spearman"][1:, 1], -1.0, "rolling w_exceeds -")
    _assert_nan(outs["pearson"][:2, 2], "rolling w_exceeds hole")
    _assert_close(outs["pearson"][3:, 2], 1.0, "rolling w_exceeds recovered")


def _summary_cases(cases: list[tuple[str, ...]], rng: np.random.Generator) -> None:
    def put(name: str, series: np.ndarray, lags: int) -> np.ndarray:
        series = np.asarray(series, dtype=np.float64).reshape(-1, 1)
        expected = summary_stats(series, lags)
        add(cases, f"summary_{name}", "summary", [("lags", lags)], series, None, expected)
        return expected

    mixed = np.array(
        [0.4, -0.2, 0.1, 0.0, 0.5, -0.3, 0.2, 0.05, -0.1, 0.3, 0.0, 0.15],
        dtype=np.float64,
    )
    got = put("lags0", mixed, 0)
    if not (got[7] == 12.0 and math.isfinite(got[4]) and 0.0 < got[6] < 1.0):
        raise AssertionError(f"summary_lags0 {got}")
    if abs(got[3] - got[4]) <= 1e-8:
        raise AssertionError("summary_lags0: ordinary t and Newey-West t coincide")

    shock = rng.normal(size=24)
    series = np.empty(24, dtype=np.float64)
    series[0] = shock[0] - 0.1
    for t in range(1, 24):
        series[t] = 0.65 * series[t - 1] + shock[t] - 0.08
    got = put("lags3", series, 3)
    if got[7] != 24.0 or not math.isfinite(got[4]) or not math.isfinite(got[5]):
        raise AssertionError(f"summary_lags3 {got}")
    if got[4] >= 0.0:
        raise AssertionError("summary_lags3: expected a negative Newey-West t")

    spotted = np.array(
        [0.2, np.nan, -0.1, 0.4, np.nan, np.nan, 0.0, 0.3, -0.4, 0.1, np.nan, 0.5, -0.2, 0.2],
        dtype=np.float64,
    )
    got = put("nans", spotted, 2)
    if got[7] != 10.0 or not (0.0 < got[6] < 1.0) or not math.isfinite(got[4]):
        raise AssertionError(f"summary_nans {got}")

    got = put("empty", np.array([np.nan, np.nan, np.nan]), 4)
    if got[7] != 0.0 or np.any(np.isfinite(got[:7])):
        raise AssertionError(f"summary_empty {got}")

    got = put("single", np.array([np.nan, -2.5, np.nan]), 3)
    if got[0] != -2.5 or got[6] != 0.0 or got[7] != 1.0 or np.any(np.isfinite(got[1:6])):
        raise AssertionError(f"summary_single {got}")

    got = put("constant", np.full(8, 5.0), 3)
    if got[0] != 5.0 or got[1] != 0.0 or got[6] != 1.0 or got[7] != 8.0:
        raise AssertionError(f"summary_constant {got}")
    if np.any(np.isfinite(got[2:6])):
        raise AssertionError(f"summary_constant statistics {got}")

    shock = rng.normal(size=18)
    positive = np.empty(18, dtype=np.float64)
    positive[0] = abs(shock[0]) + 0.2
    for t in range(1, 18):
        positive[t] = 0.5 * positive[t - 1] + abs(shock[t]) + 0.05
    got = put("all_positive", positive, 2)
    if got[6] != 1.0 or got[7] != 18.0 or not (got[0] > 0.0 and math.isfinite(got[3]) and math.isfinite(got[4])):
        raise AssertionError(f"summary_all_positive {got}")

    # lags past the sample: kernel weights still depend on ``lags``, gammas past n-1 are 0.
    long = np.array([0.5, -0.2, 0.3, 0.1, -0.4, 0.2, 0.05], dtype=np.float64)
    got = put("long_lags", long, 12)
    if got[7] != 7.0 or not math.isfinite(got[4]):
        raise AssertionError(f"summary_long_lags {got}")


def _self_check() -> None:
    sample = np.array([[1.0, 2.0], [3.0, 4.0]])
    if fmt_vec(sample) != "1.0,3.0,2.0,4.0":
        raise AssertionError("column-major layout")
    if fmt_float(float("nan")) != "nan" or fmt_float(float("inf")) != "inf":
        raise AssertionError("non-finite formatting")
    if fmt_float(float("-inf")) != "-inf" or fmt_params(None) != "-":
        raise AssertionError("formatting")

    rng = np.random.default_rng(0)
    probe = _prices(rng, 9, 3)
    probe[2, 0] = np.nan
    probe[4, 1] = np.inf
    probe[5, 2] = -np.inf
    probe[6, 0] = 0.0
    for h, lag in ((1, 0), (3, 1), (1, 2), (8, 0)):
        pandas_out = forward_returns(probe, h, lag)
        loop_out = _forward_reference(probe, h, lag)
        if not np.allclose(pandas_out, loop_out, rtol=0.0, atol=0.0, equal_nan=True):
            raise AssertionError(f"forward_returns diverged from the index formula h={h} lag={lag}")

    known = np.array([1.0, 2.0, 3.0, 4.0])
    lags0 = summary_stats(known, 0)
    lags1 = summary_stats(known, 1)
    if abs(lags0[0] - 2.5) > 1e-12 or abs(lags0[4] - 4.47213595499958) > 1e-9:
        raise AssertionError(f"summary lags0 {lags0}")
    if abs(lags1[4] - 4.0) > 1e-12:
        raise AssertionError(f"summary lags1 {lags1}")
    # Ordinary t uses the sample standard deviation; lags=0 Newey-West uses /n.
    std = math.sqrt(5.0 / 3.0)
    if abs(lags0[1] - std) > 1e-12 or abs(lags0[3] - (2.5 / std) * 2.0) > 1e-12:
        raise AssertionError(f"summary ordinary t {lags0}")
    if abs(lags0[5] - math.erfc(abs(lags0[4]) / math.sqrt(2.0))) > 1e-12:
        raise AssertionError("summary p-value")

    empty = summary_stats(np.array([np.nan, np.nan]), 3)
    if empty[7] != 0.0 or np.any(np.isfinite(empty[:7])):
        raise AssertionError("empty summary")
    single = summary_stats(np.array([-2.5]), 5)
    if single[0] != -2.5 or single[7] != 1.0 or np.any(np.isfinite(single[1:6])):
        raise AssertionError("single summary")
    constant = summary_stats(np.full(6, 5.0), 3)
    if constant[1] != 0.0 or np.any(np.isfinite(constant[2:6])):
        raise AssertionError("constant summary")
    hits = summary_stats(np.array([1.0, np.nan, 0.0, np.nan, -1.0, 2.0]), 0)
    if hits[6] != 0.5 or hits[7] != 4.0:
        raise AssertionError("hit rate after compacting NaNs")
    summary_stats(np.random.default_rng(1).normal(size=6) + 0.2, lags=10)

    tied_x = np.array([1.0, 1.0, 2.0, 3.0])
    tied_y = np.array([1.0, 2.0, 3.0, 4.0])
    average = scipy.stats.pearsonr(
        scipy.stats.rankdata(tied_x, method="average"),
        scipy.stats.rankdata(tied_y, method="average"),
    )[0]
    minimum = scipy.stats.pearsonr(
        scipy.stats.rankdata(tied_x, method="min"),
        scipy.stats.rankdata(tied_y, method="min"),
    )[0]
    spearman = _correlate(tied_x, tied_y, min_obs=4, method="spearman")
    if abs(spearman - float(average)) > 1e-12 or abs(spearman - float(minimum)) < 1e-6:
        raise AssertionError(f"average ranks {spearman} vs {average} / {minimum}")
    if math.isfinite(_correlate(np.ones(4), tied_y, min_obs=4, method="pearson")):
        raise AssertionError("constant pearson")
    if math.isfinite(_correlate(np.ones(4), tied_y, min_obs=4, method="spearman")):
        raise AssertionError("constant spearman")

    perfect = np.array([1.0, 2.0, 4.0, 8.0])
    if abs(_correlate(perfect, 3.0 * perfect + 1.0, 4, "pearson") - 1.0) > 1e-12:
        raise AssertionError("perfect pearson")
    if abs(_correlate(perfect, -3.0 * perfect + 1.0, 4, "spearman") + 1.0) > 1e-12:
        raise AssertionError("anti spearman")

    kept_x = np.array([[1.0, 1.0, 1.0, 9.0]])
    kept_y = np.array([[2.0, 3.0, 4.0, np.nan]])
    if math.isfinite(cs_corr(kept_x, kept_y, 3, "pearson")[0]):
        raise AssertionError("constant on the kept subset")
    inf_x = np.array([[1.0, 2.0, np.inf, 4.0]])
    inf_y = np.array([[2.0, 4.0, 3.0, 8.0]])
    if abs(cs_corr(inf_x, inf_y, 3, "pearson")[0] - 1.0) > 1e-12:
        raise AssertionError("inf was not dropped")

    grow_x = np.array([[1.0], [2.0], [3.0]])
    grow_y = np.array([[2.0], [4.0], [6.0]])
    grown = rolling_corr(grow_x, grow_y, window=10, min_obs=2, method="pearson")
    if not math.isnan(grown[0, 0]) or abs(grown[1, 0] - 1.0) > 1e-12 or abs(grown[2, 0] - 1.0) > 1e-12:
        raise AssertionError(f"expanding window {grown[:, 0]}")
    one = rolling_corr(grow_x, grow_y, window=1, min_obs=1, method="spearman")
    if np.any(np.isfinite(one)):
        raise AssertionError("w=1 should be NaN")


def build_cases() -> list[tuple[str, ...]]:
    rng = np.random.default_rng(SEED)
    cases: list[tuple[str, ...]] = []
    _forward_cases(cases, rng)
    _cs_cases(cases, rng)
    _ts_cases(cases, rng)
    _rolling_cases(cases, rng)
    _summary_cases(cases, rng)
    names = [row[0] for row in cases]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate case name")
    if len(cases) < 40:
        raise AssertionError(f"only {len(cases)} cases")
    return cases


def render(cases: list[tuple[str, ...]]) -> str:
    lines = ["\t".join(HEADER)]
    lines.extend("\t".join(row) for row in cases)
    return "\n".join(lines) + "\n"


def main() -> None:
    _self_check()
    first = render(build_cases())
    second = render(build_cases())
    if first != second:
        raise AssertionError("generator is not deterministic")
    if "\r" in first:
        raise AssertionError("fixture contains CR")
    first.encode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(first, encoding="utf-8", newline="\n")
    print(f"wrote {first.count(chr(10)) - 1} cases to {OUT}")


if __name__ == "__main__":
    main()
