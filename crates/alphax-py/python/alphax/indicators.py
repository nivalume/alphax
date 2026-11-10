"""Indicators, swept over parameter grids, with parameter-labelled columns.

The eight built-in indicators run in Rust. Any other function becomes one
with ``IndicatorFactory``, which runs it per parameter combination and lays
the outputs out the same way; ``from_talib``, ``from_pandas_ta``, ``from_ta``
and ``from_smc`` build factories from those libraries' functions, imported
when used.

Every result is an ``IndicatorResult``: one ``(n_rows, n_combos * n_input)``
array per output, combination-major (every input column under the first
combination, then the next), and ``columns``, each parameter's value per
output column, so ``result.select(window=20)`` picks a combination::

    r = alphax.indicators.ma(close, window=[10, 20, 50], ewm=[False, True])
    r.ma.shape            # (n_rows, 6 * n_cols)
    r.columns["window"]   # [10, 10, ..., 50] per output column
    fast = r.select(window=10, ewm=False).ma

Parameters given as lists are combined as a product; ``product=False``
pairs them up instead, position by position.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import _alphax

__all__ = [
    "Stream",
    "IndicatorFactory",
    "IndicatorResult",
    "atr",
    "bbands",
    "from_pandas_ta",
    "from_smc",
    "from_ta",
    "from_talib",
    "ma",
    "macd",
    "mstd",
    "obv",
    "rsi",
    "stoch",
    "true_range",
]


@dataclass
class IndicatorResult:
    """An indicator's outputs by name, and each output column's labels.

    ``columns`` maps each parameter (and ``"input"``, the input column) to
    one value per output column. Outputs are also attributes:
    ``result.ma``.
    """

    outputs: dict[str, np.ndarray]
    columns: dict[str, list] = field(default_factory=dict)

    def __getattr__(self, name: str) -> np.ndarray:
        outputs = self.__dict__.get("outputs", {})
        if name in outputs:
            return outputs[name]
        raise AttributeError(f"no output {name!r}; outputs are {list(outputs)}")

    def select(self, **values: Any) -> "IndicatorResult":
        """The columns whose parameters equal ``values``."""
        n = next(iter(self.outputs.values())).shape[1]
        keep = np.ones(n, dtype=bool)
        for name, value in values.items():
            if name not in self.columns:
                raise KeyError(f"no parameter {name!r}; parameters are {list(self.columns)}")
            keep &= np.array([v == value for v in self.columns[name]])
        idx = np.flatnonzero(keep)
        return IndicatorResult(
            {k: v[:, idx] for k, v in self.outputs.items()},
            {k: [v[i] for i in idx] for k, v in self.columns.items()},
        )

    def params(self) -> list[dict]:
        """Each output column's parameters, as a dict."""
        names = list(self.columns)
        return [dict(zip(names, vals)) for vals in zip(*self.columns.values())]


def _matrix(a: Any, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got {a.ndim} dimensions")
    return np.asfortranarray(a)


def _grid(product: bool, **params: Any) -> dict[str, list]:
    lists = {k: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] for k, v in params.items()}
    if product:
        combos = list(itertools.product(*lists.values()))
    else:
        n = max(len(v) for v in lists.values())
        for k, v in lists.items():
            if len(v) not in (1, n):
                raise ValueError(f"{k} has {len(v)} values; with product=False each has 1 or {n}")
        combos = [tuple(v[i if len(v) > 1 else 0] for v in lists.values()) for i in range(n)]
    names = list(lists)
    return {name: [c[j] for c in combos] for j, name in enumerate(names)}


def _result(raw: dict) -> IndicatorResult:
    columns = dict(raw["columns"])
    if "symbol" in columns:
        columns["input"] = columns.pop("symbol")
    return IndicatorResult(dict(raw["outputs"]), columns)


def ma(close, window=14, ewm=False, *, adjust=True, product=True) -> IndicatorResult:
    """Moving average: simple, or exponential with ``ewm``. Output ``ma``."""
    g = _grid(product, window=window, ewm=ewm)
    return _result(_alphax.ind_ma(_matrix(close, "close"), [int(w) for w in g["window"]],
                                  [bool(e) for e in g["ewm"]], adjust))


def mstd(close, window=14, ewm=False, *, adjust=True, ddof=1, product=True) -> IndicatorResult:
    """Moving standard deviation. Output ``mstd``."""
    g = _grid(product, window=window, ewm=ewm)
    return _result(_alphax.ind_mstd(_matrix(close, "close"), [int(w) for w in g["window"]],
                                    [bool(e) for e in g["ewm"]], adjust, ddof))


def bbands(close, window=20, ewm=False, alpha=2.0, *, adjust=True, ddof=0, product=True) -> IndicatorResult:
    """Bollinger bands: ``middle``, and ``upper`` and ``lower`` at ``alpha``
    standard deviations from it."""
    g = _grid(product, window=window, ewm=ewm, alpha=alpha)
    return _result(_alphax.ind_bbands(_matrix(close, "close"), [int(w) for w in g["window"]],
                                      [bool(e) for e in g["ewm"]], [float(a) for a in g["alpha"]],
                                      adjust, ddof))


def rsi(close, window=14, ewm=False, *, adjust=True, product=True) -> IndicatorResult:
    """Relative strength index, 0 to 100. Output ``rsi``."""
    g = _grid(product, window=window, ewm=ewm)
    return _result(_alphax.ind_rsi(_matrix(close, "close"), [int(w) for w in g["window"]],
                                   [bool(e) for e in g["ewm"]], adjust))


def stoch(high, low, close, k_window=14, d_window=3, d_ewm=False, *, adjust=True, product=True) -> IndicatorResult:
    """Stochastic oscillator: ``percent_k`` and its average ``percent_d``."""
    g = _grid(product, k_window=k_window, d_window=d_window, d_ewm=d_ewm)
    return _result(_alphax.ind_stoch(_matrix(high, "high"), _matrix(low, "low"), _matrix(close, "close"),
                                     [int(w) for w in g["k_window"]], [int(w) for w in g["d_window"]],
                                     [bool(e) for e in g["d_ewm"]], adjust))


def macd(close, fast_window=12, slow_window=26, signal_window=9, macd_ewm=False, signal_ewm=False,
         *, adjust=True, product=True) -> IndicatorResult:
    """Moving average convergence divergence: ``macd`` and its average
    ``signal``; the histogram is their difference."""
    g = _grid(product, fast_window=fast_window, slow_window=slow_window, signal_window=signal_window,
              macd_ewm=macd_ewm, signal_ewm=signal_ewm)
    return _result(_alphax.ind_macd(
        _matrix(close, "close"), [int(w) for w in g["fast_window"]], [int(w) for w in g["slow_window"]],
        [int(w) for w in g["signal_window"]], [bool(e) for e in g["macd_ewm"]],
        [bool(e) for e in g["signal_ewm"]], adjust))


def atr(high, low, close, window=14, ewm=False, *, adjust=True, product=True) -> IndicatorResult:
    """Average true range: ``tr`` and ``atr``."""
    g = _grid(product, window=window, ewm=ewm)
    return _result(_alphax.ind_atr(_matrix(high, "high"), _matrix(low, "low"), _matrix(close, "close"),
                                   [int(w) for w in g["window"]], [bool(e) for e in g["ewm"]], adjust))


def obv(close, volume) -> IndicatorResult:
    """On-balance volume. Output ``obv``."""
    return _result(_alphax.ind_obv(_matrix(close, "close"), _matrix(volume, "volume")))


def true_range(high, low, close) -> np.ndarray:
    """Each bar's true range: the high-low range widened to the previous
    close."""
    return _alphax.ind_true_range(_matrix(high, "high"), _matrix(low, "low"), _matrix(close, "close"))


_STREAM_DEFAULTS = {
    "ma": dict(window=14, ewm=False, adjust=True),
    "mstd": dict(window=14, ewm=False, adjust=True, ddof=1),
    "bbands": dict(window=20, ewm=False, alpha=2.0, adjust=True, ddof=0),
    "rsi": dict(window=14, ewm=False, adjust=True),
    "stoch": dict(k_window=14, d_window=3, d_ewm=False, adjust=True),
    "macd": dict(fast_window=12, slow_window=26, signal_window=9, macd_ewm=False, signal_ewm=False, adjust=True),
    "atr": dict(window=14, ewm=False, adjust=True),
    "obv": dict(),
}


class Stream:
    """One indicator over ``n_cols`` series, a bar at a time, for live use.

    Parameters are the batch function's, one value each, with the same
    defaults. Each :meth:`update` takes one bar for every column and returns
    that bar's outputs, equal bit for bit to the batch function's on the
    whole history::

        s = alphax.indicators.Stream("rsi", n_cols=3, window=14)
        for bar in live_bars:
            out = s.update(bar.close)        # {"rsi": array of 3}
        s = alphax.indicators.Stream("atr", n_cols=3)
        s.update(close, high=high, low=low)  # {"tr": ..., "atr": ...}
    """

    def __init__(self, kind: str, n_cols: int = 1, **params: Any):
        if kind not in _STREAM_DEFAULTS:
            raise ValueError(f"no streaming indicator {kind!r}; there are {', '.join(_STREAM_DEFAULTS)}")
        unknown = set(params) - set(_STREAM_DEFAULTS[kind])
        if unknown:
            raise TypeError(f"{kind} takes no parameter {sorted(unknown)[0]!r}")
        self.kind = kind
        self.params = {**_STREAM_DEFAULTS[kind], **params}
        self._inner = _alphax._IndicatorStream(kind, n_cols, self.params)

    @property
    def names(self) -> list[str]:
        return list(self._inner.names)

    def update(self, close, high=None, low=None, volume=None) -> dict[str, np.ndarray]:
        def row(x):
            return None if x is None else np.atleast_1d(np.asarray(x, dtype=float)).tolist()
        return dict(self._inner.update(row(close), row(high), row(low), row(volume)))


class IndicatorFactory:
    """Any function as a swept indicator.

    ``func(*inputs, **params)`` takes the inputs one column at a time as 1-D
    arrays and returns one array (``outputs`` of length 1) or a tuple of
    them, each as long as the inputs. ``run`` calls it for every parameter
    combination and input column and lays the results out as the built-in
    indicators do::

        def zscore(close, window):
            m = alphax.indicators.ma(close, window).ma[:, 0]
            s = alphax.indicators.mstd(close, window).mstd[:, 0]
            return (close - m) / s

        Z = alphax.indicators.IndicatorFactory(zscore, inputs=["close"],
                                               params=["window"], outputs=["z"])
        r = Z.run(close, window=[10, 20])

    ``defaults`` gives parameters a value when ``run`` does not.
    """

    def __init__(self, func: Callable, *, inputs: Sequence[str], params: Sequence[str] = (),
                 outputs: Sequence[str] = ("out",), defaults: dict | None = None):
        self.func = func
        self.inputs = list(inputs)
        self.params = list(params)
        self.outputs = list(outputs)
        self.defaults = dict(defaults or {})

    def run(self, *inputs: Any, product: bool = True, **params: Any) -> IndicatorResult:
        if len(inputs) != len(self.inputs):
            raise TypeError(f"expected {len(self.inputs)} inputs ({', '.join(self.inputs)}); got {len(inputs)}")
        unknown = set(params) - set(self.params)
        if unknown:
            raise TypeError(f"unknown parameters {sorted(unknown)}; parameters are {self.params}")
        values = {p: params.get(p, self.defaults.get(p)) for p in self.params}
        missing = [p for p, v in values.items() if v is None]
        if missing:
            raise TypeError(f"no value for {missing}")
        arrays = [_matrix(a, n) for a, n in zip(inputs, self.inputs)]
        n_rows, n_cols = arrays[0].shape
        if any(a.shape != (n_rows, n_cols) for a in arrays):
            raise ValueError("every input must be the same shape")
        g = _grid(product, **values) if self.params else {}
        n_combos = len(next(iter(g.values()))) if g else 1
        out = {name: np.full((n_rows, n_combos * n_cols), np.nan, order="F") for name in self.outputs}
        for i in range(n_combos):
            kw = {p: g[p][i] for p in self.params}
            for c in range(n_cols):
                got = self.func(*(a[:, c] for a in arrays), **kw)
                got = got if isinstance(got, tuple) else (got,)
                if len(got) != len(self.outputs):
                    raise ValueError(f"func returned {len(got)} outputs; expected {len(self.outputs)}")
                for name, v in zip(self.outputs, got):
                    v = np.asarray(v, dtype=np.float64)
                    if v.shape != (n_rows,):
                        raise ValueError(f"output {name!r} has shape {v.shape}; expected ({n_rows},)")
                    out[name][:, i * n_cols + c] = v
        columns = {p: [g[p][i] for i in range(n_combos) for _ in range(n_cols)] for p in self.params}
        columns["input"] = list(range(n_cols)) * n_combos
        return IndicatorResult(out, columns)


def _import(module: str, extra: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"{module} is not installed; pip install {extra}") from e


def from_talib(name: str, *, inputs: Sequence[str] = ("close",), params: Sequence[str] = (),
               outputs: Sequence[str] | None = None, defaults: dict | None = None) -> IndicatorFactory:
    """A TA-Lib function (``"RSI"``, ``"BBANDS"``...) as a factory. TA-Lib's
    names its parameters (``timeperiod``...); list the ones to sweep."""
    talib = _import("talib", "TA-Lib")
    func = getattr(talib, name)
    outputs = list(outputs or ["out"])

    def run(*cols, **kw):
        got = func(*cols, **kw)
        return tuple(got) if isinstance(got, (tuple, list)) else got

    return IndicatorFactory(run, inputs=inputs, params=params, outputs=outputs, defaults=defaults)


def _pandas_runner(call: Callable, inputs: Sequence[str], outputs: Sequence[str] | None):
    pd = _import("pandas", "pandas")

    def run(*cols, **kw):
        series = {n: pd.Series(c) for n, c in zip(inputs, cols)}
        got = call(series, **kw)
        if isinstance(got, pd.DataFrame):
            cols_out = list(outputs) if outputs else list(got.columns)
            return tuple(got[c].to_numpy(dtype=float) for c in cols_out)
        return np.asarray(got, dtype=float)

    return run


def from_pandas_ta(name: str, *, inputs: Sequence[str] = ("close",), params: Sequence[str] = (),
                   outputs: Sequence[str] | None = None, defaults: dict | None = None) -> IndicatorFactory:
    """A pandas-ta function (``"rsi"``, ``"bbands"``...) as a factory.
    ``outputs`` names the DataFrame columns to keep for a multi-output
    indicator."""
    ta = _import("pandas_ta", "pandas-ta")
    func = getattr(ta, name)
    run = _pandas_runner(lambda s, **kw: func(**s, **kw), inputs, outputs)
    return IndicatorFactory(run, inputs=inputs, params=params, outputs=list(outputs or ["out"]),
                            defaults=defaults)


def from_ta(cls_path: str, method: str, *, inputs: Sequence[str] = ("close",), params: Sequence[str] = (),
            outputs: Sequence[str] = ("out",), defaults: dict | None = None) -> IndicatorFactory:
    """A ``ta`` library indicator: ``cls_path`` is its class
    (``"momentum.RSIIndicator"``) and ``method`` the output method
    (``"rsi"``); the class takes the inputs by name."""
    ta = _import("ta", "ta")
    module, name = cls_path.rsplit(".", 1)
    cls = getattr(getattr(ta, module), name)
    run = _pandas_runner(lambda s, **kw: getattr(cls(**s, **kw), method)(), inputs, None)
    return IndicatorFactory(run, inputs=inputs, params=params, outputs=list(outputs), defaults=defaults)


def from_smc(name: str, *, outputs: Sequence[str], params: Sequence[str] = (),
             defaults: dict | None = None) -> IndicatorFactory:
    """A smartmoneyconcepts function (``"fvg"``, ``"swing_highs_lows"``...)
    as a factory over ``open``, ``high``, ``low``, ``close``, ``volume``;
    ``outputs`` names the DataFrame columns to keep."""
    smc = _import("smartmoneyconcepts", "smartmoneyconcepts").smc
    func = getattr(smc, name)
    pd = _import("pandas", "pandas")
    inputs = ["open", "high", "low", "close", "volume"]

    def call(series, **kw):
        return func(pd.DataFrame(series), **kw)

    run = _pandas_runner(call, inputs, outputs)
    return IndicatorFactory(run, inputs=inputs, params=params, outputs=list(outputs), defaults=defaults)
