"""phasor — a vectorized backtesting engine, in Rust.

The shell is deliberately thin. Everything that computes lives in Rust; this
module exists to do the two things a Rust extension cannot do for the caller:
put arrays in the layout the engine reads, and give back something a notebook
can print.

    import numpy as np, phasor

    close = np.array([[10.0], [11.0], [12.0], [11.0]])
    entries = np.array([[True], [False], [False], [False]])
    exits = np.array([[False], [False], [True], [False]])

    pf = phasor.from_signals(close, entries, exits, init_cash=100.0, fees=0.001)
    print(pf.total_return())     # one value per column
    print(pf.stats())            # the table, for column 0

Layout
------
The engine is column-major; NumPy is row-major by default. Every array is
passed through ``np.asfortranarray``, which is free when the array is already
in that order and a visible copy when it is not. The Rust side refuses a
C-ordered array rather than reading it as a transposed price series, so a
caller who reaches past this module gets an error instead of a plausible
backtest on scrambled data.

Shapes
------
A column is one symbol under one parameter set. By default columns have
independent cash. ``cash_sharing=True`` groups adjacent columns into accounts;
``threads`` parallelizes accounts. A 1-D array of prices is read as one column.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import (
    _phasor,
    _timing,
    allocation,
    cache,
    chunked,
    config,
    data,
    distributed,
    docs,
    events,
    frames,
    futures,
    fx,
    indicators,
    labels,
    params,
    patterns,
    plotting,
    portfolios,
    predictions,
    processors,
    rates,
    resolution,
    returns,
    robust,
    schedule,
    sessions,
    signals,
    synthetic,
    tools,
    validation,
)
from ._phasor import InstrumentId, BarType
from ._phasor import Context, Portfolio, Segment, SimState

__all__ = [
    "Expression",
    "FactorGrid",
    "FactorSet",
    "Context",
    "Portfolio",
    "Segment",
    "SimState",
    "Stepper",
    "alpha158",
    "alpha158_features",
    "alpha360",
    "qlib_label",
    "wq101",
    "wq101_formula",
    "data",
    "events",
    "frames",
    "futures",
    "fx",
    "indicators",
    "labels",
    "patterns",
    "plotting",
    "portfolios",
    "predictions",
    "processors",
    "rates",
    "resolution",
    "returns",
    "robust",
    "sessions",
    "signals",
    "synthetic",
    "validation",
    "InstrumentId",
    "BarType",
    "from_order_func",
    "from_orders",
    "load",
    "save",
    "from_signals",
    "strategy",
    "factors",
    "crossed_above",
]
__version__ = "0.2.0"


def _matrix(value: Any, name: str) -> np.ndarray:
    """A 2-D, F-ordered, float64 array, whatever shape came in.

    A 1-D input is one column, not one row. That is the direction that matches
    "a column is a run": a bare price series is a single backtest, not a single
    bar across many symbols.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got {array.ndim} dimensions")
    return np.asfortranarray(array)


def _factor_fields(
    fields: Mapping[str, Any], shape: tuple[int, int] | None = None
) -> tuple[list[tuple[str, np.ndarray]], tuple[int, int]]:
    """The fields as F-ordered float64 matrices, and their common shape.
    ``shape`` is what ``index`` and ``symbols`` imply, which a set that reads
    only point-in-time data needs in place of fields."""
    if not isinstance(fields, Mapping):
        raise TypeError("factor inputs must be a mapping of field names to arrays")
    implied = shape
    shape = None
    prepared = []
    for name, value in fields.items():
        if not isinstance(name, str):
            raise TypeError("factor field names must be strings")
        array = _matrix(value, f"factor field {name!r}")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"factor field {name!r} has shape {array.shape}, expected {shape}"
            )
        prepared.append((name, array))
    if shape is None:
        if implied is None:
            raise ValueError(
                "factor inputs cannot be empty: pass fields, or index and symbols "
                "for a set that reads only point-in-time data"
            )
        shape = implied
    elif implied is not None and tuple(implied) != tuple(shape):
        raise ValueError(f"index and symbols imply shape {implied}, but the fields are {shape}")
    return prepared, tuple(shape)


def _timestamps(index: Any) -> np.ndarray:
    """Wall-clock nanoseconds of each row. A timezone-aware pandas index is
    read on its own clock: 09:30 in Shanghai stays 09:30."""
    values = index
    if getattr(values, "tz", None) is not None:
        values = values.tz_localize(None)
    elif getattr(getattr(values, "dt", None), "tz", None) is not None:
        values = values.dt.tz_localize(None)
    array = np.asarray(values)
    if array.dtype.kind == "M":
        return np.ascontiguousarray(array.astype("datetime64[ns]").astype(np.int64))
    if array.dtype.kind in "iu":
        return np.ascontiguousarray(array.astype(np.int64))
    raise TypeError("index must hold datetimes (datetime64, a pandas DatetimeIndex) or int64 nanoseconds")


def _days(values: Any) -> np.ndarray:
    """Days since 1970-01-01 of dates given as datetimes, ``YYYY-MM-DD``
    strings, or Qlib's ``YYYYMMDD`` integers."""
    if getattr(values, "tz", None) is not None or getattr(getattr(values, "dt", None), "tz", None) is not None:
        return _timestamps(values) // 86_400_000_000_000
    array = np.asarray(values)
    if array.dtype.kind in "iu":
        array = array.astype(np.int64)
        months = (array // 10000 - 1970).astype("datetime64[Y]").astype("datetime64[M]")
        months = months + (array // 100 % 100 - 1).astype("timedelta64[M]")
        days = months.astype("datetime64[D]") + (array % 100 - 1).astype("timedelta64[D]")
        return days.astype(np.int64)
    return np.asarray(array, dtype="datetime64[D]").astype(np.int64)


def _pit_records(pit: Any, symbols: Sequence[str] | None) -> list[tuple]:
    """Point-in-time records from a long table (a DataFrame, or a mapping of
    equal-length columns) with columns ``symbol``, ``field``, ``date``,
    ``period`` and ``value``, as Qlib's normalized PIT data has. Records of
    symbols outside ``symbols`` are dropped; without ``symbols``, ``symbol``
    is the column position."""
    try:
        raw = {key: pit[key] for key in ("symbol", "field", "date", "period", "value")}
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("pit must have columns symbol, field, date, period and value") from e
    columns = {key: np.asarray(value) for key, value in raw.items()}
    if symbols is None:
        cols = columns["symbol"].astype(np.int64)
    else:
        position = {name: k for k, name in enumerate(symbols)}
        cols = np.array([position.get(str(s), -1) for s in columns["symbol"]], dtype=np.int64)
    keep = cols >= 0
    fields = np.array([str(f).lstrip("$").lower() for f in columns["field"]])
    days = _days(raw["date"])
    periods = columns["period"].astype(np.int64)
    values = columns["value"].astype(np.float64)
    records = []
    for field in dict.fromkeys(fields[keep]):
        mask = keep & (fields == field)
        records.append((str(field), cols[mask], days[mask], periods[mask], values[mask]))
    return records


def _instruments(instruments: Mapping[str, Mapping[str, Any]], n_rows: int) -> list[tuple]:
    out = []
    for name, own in instruments.items():
        arrays = []
        for field, values in own.items():
            array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
            if array.shape[0] != n_rows:
                raise ValueError(
                    f"instrument {name!r} field {field!r} has {array.shape[0]} rows, expected {n_rows}"
                )
            arrays.append((str(field), array))
        out.append((str(name), arrays))
    return out


def _factor_context(
    fields: Mapping[str, Any],
    index: Any,
    symbols: Sequence[Any] | None,
    instruments: Mapping[str, Mapping[str, Any]] | None,
    pit: Any,
) -> tuple[list[tuple[str, np.ndarray]], tuple[int, int], dict]:
    names = None if symbols is None else [str(s) for s in symbols]
    implied = None if index is None or names is None else (len(index), len(names))
    prepared, shape = _factor_fields(fields, implied)
    context: dict[str, Any] = {}
    if index is not None:
        timestamps = _timestamps(index)
        if timestamps.shape[0] != shape[0]:
            raise ValueError(f"index has {timestamps.shape[0]} rows, expected {shape[0]}")
        context["timestamps"] = timestamps
    if names is not None:
        context["symbols"] = names
    if instruments:
        context["instruments"] = _instruments(instruments, shape[0])
    if pit is not None:
        context["pit"] = _pit_records(pit, names)
    return prepared, shape, context


class FactorSet:
    """A compiled, reusable set of named factor expressions."""

    __slots__ = ("_inner",)

    def __init__(self, inner: _phasor._FactorSet):
        self._inner = inner

    @classmethod
    def compile(cls, factors: Mapping[str, str]) -> FactorSet:
        if not isinstance(factors, Mapping):
            raise TypeError("factors must be a mapping of names to expressions")
        pairs = []
        for name, source in factors.items():
            if not isinstance(name, str) or not isinstance(source, str):
                raise TypeError("factor names and expressions must be strings")
            pairs.append((name, source))
        return cls(_phasor._FactorSet.compile(pairs))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._inner.names)

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(self._inner.required_fields)

    @property
    def uses_future(self) -> bool:
        return self._inner.uses_future

    @property
    def instrument_names(self) -> tuple[str, ...]:
        """Instruments read with ``ChangeInstrument`` (or Qlib's
        ``Mask(e, "X")``): symbols of the panel, or ``instruments=``."""
        return tuple(self._inner.instrument_names)

    @property
    def needs_timestamps(self) -> bool:
        """Whether evaluation needs ``index=``: ``DayLast``, ``DayCumsum``,
        ``Date``, ``P`` and ``PRef`` group and compare by the bar times."""
        return self._inner.needs_timestamps

    @property
    def lookbacks(self) -> dict[str, int]:
        """Per factor, how many rows before the current one its windows and
        lags reach: ``Mean($close, 20)`` is 19, ``Ref(Mean($close, 5), 2)`` is
        6. Its value is complete from that many rows after its inputs begin."""
        return dict(zip(self.names, self._inner.lookbacks))

    @property
    def lookback(self) -> int:
        """The longest of :attr:`lookbacks`."""
        return max(self._inner.lookbacks, default=0)

    @property
    def warmup(self) -> bool:
        """Whether :meth:`with_warmup` is on."""
        return self._inner.warmup

    def with_warmup(self, on: bool = True) -> FactorSet:
        """The same set, evaluating to ``NaN`` on each symbol's rows before a
        factor's windows have filled: until its input fields all have values
        for that symbol, plus its lookback. Cross-sectional operators then
        leave out a symbol that is still warming up.

        Qlib's operators (and this set by default) give a value from the first
        row, from a partial window; in a backtest those rows trade on a 2-bar
        standard deviation. Turn this on for signals."""
        return FactorSet(self._inner.with_warmup(bool(on)))

    def evaluate(
        self,
        fields: Mapping[str, Any],
        *,
        index: Any = None,
        symbols: Sequence[Any] | None = None,
        instruments: Mapping[str, Mapping[str, Any]] | None = None,
        pit: Any = None,
    ) -> dict[str, np.ndarray]:
        """Every factor over ``fields`` (each ``(rows, symbols)``).

        ``index``: the bar times, one per row (datetime64, a pandas
        DatetimeIndex, or int64 nanoseconds), read on the market's own clock.
        ``symbols``: the column names, among which ``ChangeInstrument("X",
        ...)`` looks ``X`` up. ``instruments``: instruments outside the panel
        such as a benchmark, ``{name: {field: 1-D array}}``. ``pit``:
        point-in-time records for ``$$field``, a long table with columns
        ``symbol``, ``field``, ``date``, ``period`` and ``value``.
        """
        prepared, shape, context = _factor_context(fields, index, symbols, instruments, pit)
        return {
            name: values.reshape(shape, order="F")
            for name, values in self._inner.evaluate(prepared, shape, **context).items()
        }

    def evaluate_batches(
        self,
        fields: Mapping[str, Any],
        batch: int,
        sink: Callable[[int, int, dict], Any],
        *,
        index: Any = None,
        symbols: Sequence[Any] | None = None,
        instruments: Mapping[str, Mapping[str, Any]] | None = None,
        pit: Any = None,
    ) -> None:
        """Evaluates ``batch`` symbols (columns) at a time and calls
        ``sink(start, stop, values)`` for each, ``values`` shaped
        ``(rows, stop - start)``. Without a cross-sectional operator only one
        batch is in memory at a time; with one, the whole panel is evaluated
        first. The keyword arguments are :meth:`evaluate`'s."""
        prepared, shape, context = _factor_context(fields, index, symbols, instruments, pit)

        def forward(start: int, stop: int, values: dict) -> None:
            sink(start, stop, {k: v.reshape((shape[0], stop - start), order="F") for k, v in values.items()})

        self._inner.evaluate_batches(prepared, shape, int(batch), forward, **context)

    def to_parquet(self, path: str, fields: Mapping[str, Any], batch: int = 256,
                   index: Any = None, symbols: Sequence[Any] | None = None,
                   instruments: Mapping[str, Mapping[str, Any]] | None = None,
                   pit: Any = None) -> None:
        """Writes every factor to a Parquet file, one row group per batch of
        symbols: long format, columns ``symbol``, ``row`` (or ``time`` from
        ``index``) and one per factor. ``index`` also serves as the bar times
        and ``symbols`` as the names :meth:`evaluate` reads. Needs pyarrow."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("FactorSet.to_parquet needs pyarrow: pip install pyarrow") from e
        writer = None
        n_rows = None

        def sink(start: int, stop: int, values: dict) -> None:
            nonlocal writer, n_rows
            n_rows = next(iter(values.values())).shape[0]
            labels = list(range(start, stop)) if symbols is None else list(symbols[start:stop])
            columns = {"symbol": np.repeat(np.asarray(labels), n_rows)}
            rows = np.tile(np.arange(n_rows), stop - start)
            columns["row" if index is None else "time"] = rows if index is None else np.asarray(index)[rows]
            for name, v in values.items():
                columns[name] = v.reshape(-1, order="F")
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)

        reads_symbols = bool(self.instrument_names) or pit is not None
        try:
            self.evaluate_batches(
                fields,
                batch,
                sink,
                index=index if self.needs_timestamps else None,
                symbols=symbols if reads_symbols else None,
                instruments=instruments,
                pit=pit,
            )
        finally:
            if writer is not None:
                writer.close()


def _param_text(value: Any) -> str:
    """A parameter value as expression source: numbers round-trip exactly."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


class FactorGrid:
    """Factor expressions with ``{name}`` placeholders, compiled for every
    case of a parameter grid into one graph::

        grid = phasor.FactorGrid.compile(
            {"z": "($close - Mean($close, {n})) / Std($close, {n})"}, n=[10, 20, 50])
        z = grid.evaluate({"close": close})["z"]        # (rows, 3 * symbols)
        pf = phasor.from_signals(close, z > 1, z < 0, column_map=grid.column_map(close.shape[1]))

    A placeholder can stand anywhere: a window, a threshold, a field
    (``${field}``). Every case is compiled into the same graph, so what does
    not depend on a parameter is computed once, and one pass over the
    symbols evaluates every case. Each factor's columns are
    ``case * n_symbols + symbol``, the layout of phasor's sweeps; the
    keyword arguments of :meth:`compile` form the full product, or pass
    ``cases``, e.g. ``phasor.params.grid(..., where=...)``.
    """

    __slots__ = ("_inner", "_cases")

    def __init__(self, inner: Any, cases: list[dict]):
        self._inner = inner
        self._cases = cases

    @classmethod
    def compile(
        cls, factors: Mapping[str, str], cases: Sequence[Mapping[str, Any]] | None = None, **grid: Any
    ) -> FactorGrid:
        if not isinstance(factors, Mapping):
            raise TypeError("factors must be a mapping of names to expressions")
        if cases is not None and grid:
            raise TypeError("pass either cases or grid keywords, not both")
        if cases is None:
            from .params import grid as product

            cases = product(**grid) if grid else [{}]
        cases = [dict(case) for case in cases]
        pairs = [(str(name), str(source)) for name, source in factors.items()]
        texts = [[(str(k), _param_text(v)) for k, v in case.items()] for case in cases]
        return cls(_phasor._FactorGrid.compile(pairs, texts), cases)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._inner.names)

    @property
    def cases(self) -> list[dict]:
        """The parameter values of each case, in column order."""
        return [dict(case) for case in self._cases]

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(self._inner.required_fields)

    @property
    def uses_future(self) -> bool:
        return self._inner.uses_future

    @property
    def needs_timestamps(self) -> bool:
        return self._inner.needs_timestamps

    @property
    def lookbacks(self) -> dict[str, list[int]]:
        """Per factor, each case's lookback (:attr:`FactorSet.lookbacks`)."""
        return dict(zip(self.names, self._inner.lookbacks))

    @property
    def lookback(self) -> int:
        return max((max(v) for v in self._inner.lookbacks), default=0)

    @property
    def warmup(self) -> bool:
        return self._inner.warmup

    def with_warmup(self, on: bool = True) -> FactorGrid:
        """As :meth:`FactorSet.with_warmup`."""
        return FactorGrid(self._inner.with_warmup(bool(on)), self._cases)

    def column_map(self, n_symbols: int) -> np.ndarray:
        """The ``column_map`` that shares a ``(rows, n_symbols)`` panel across
        the cases: ``[0, 1, ..., n_symbols - 1]`` once per case."""
        return np.tile(np.arange(int(n_symbols)), len(self._cases))

    def evaluate(
        self,
        fields: Mapping[str, Any],
        *,
        index: Any = None,
        symbols: Sequence[Any] | None = None,
        instruments: Mapping[str, Mapping[str, Any]] | None = None,
        pit: Any = None,
    ) -> dict[str, np.ndarray]:
        """Every factor over every case, each ``(rows, n_cases * n_symbols)``.
        The keyword arguments are :meth:`FactorSet.evaluate`'s."""
        prepared, shape, context = _factor_context(fields, index, symbols, instruments, pit)
        wide = (shape[0], shape[1] * len(self._cases))
        return {
            name: values.reshape(wide, order="F")
            for name, values in self._inner.evaluate(prepared, shape, **context).items()
        }


class Expression:
    """One compiled factor expression."""

    __slots__ = ("_factors",)

    def __init__(self, factors: FactorSet):
        self._factors = factors

    @classmethod
    def compile(cls, source: str) -> Expression:
        if not isinstance(source, str):
            raise TypeError("expression must be a string")
        return cls(FactorSet.compile({"value": source}))

    @property
    def required_fields(self) -> tuple[str, ...]:
        return self._factors.required_fields

    @property
    def uses_future(self) -> bool:
        return self._factors.uses_future

    @property
    def lookback(self) -> int:
        """How many rows before the current one the expression reads."""
        return self._factors.lookback

    def with_warmup(self, on: bool = True) -> Expression:
        """As :meth:`FactorSet.with_warmup`."""
        return Expression(self._factors.with_warmup(on))

    def evaluate(self, fields: Mapping[str, Any], **context: Any) -> np.ndarray:
        """As :meth:`FactorSet.evaluate`, for the one expression."""
        return self._factors.evaluate(fields, **context)["value"]


_ALPHA158_SECTIONS = ("kbar", "price", "volume", "rolling")


def _alpha158_spec(config: Mapping[str, Any] | None) -> tuple | None:
    """Qlib's ``Alpha158DL`` config dict as the binding's plain values, with
    Qlib's defaults for the keys a present section leaves out."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("an Alpha158 config is a mapping like Qlib's Alpha158DL config")
    unknown = sorted(set(config) - set(_ALPHA158_SECTIONS))
    if unknown:
        raise ValueError(f"unknown Alpha158 config sections {unknown}; expected {list(_ALPHA158_SECTIONS)}")

    def section(key: str) -> Mapping[str, Any]:
        return config[key] or {}

    price = None
    if "price" in config:
        p = section("price")
        price = (
            [int(d) for d in p.get("windows", range(5))],
            [str(f) for f in p.get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])],
        )
    volume = None
    if "volume" in config:
        volume = [int(d) for d in section("volume").get("windows", range(5))]
    rolling = None
    if "rolling" in config:
        r = section("rolling")
        include = r.get("include")
        rolling = (
            [int(d) for d in r.get("windows", [5, 10, 20, 30, 60])],
            None if include is None else [str(k) for k in include],
            [str(k) for k in r.get("exclude", [])],
        )
    return ("kbar" in config, price, volume, rolling)


def alpha158(config: Mapping[str, Any] | None = None) -> FactorSet:
    """Qlib's Alpha158: the default 158 features, or those ``config`` selects.

    ``config`` is the dict ``Alpha158DL.get_feature_config`` takes: sections
    ``kbar``, ``price`` (``windows``, ``feature``), ``volume`` (``windows``)
    and ``rolling`` (``windows``, ``include``, ``exclude``), each on when
    present, with Qlib's defaults for keys a section leaves out. A rolling
    family is named by Qlib's key (the ``MIN`` family is ``"LOW"``)."""
    return FactorSet(_phasor.alpha158(_alpha158_spec(config)))


def alpha158_features(config: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """The ``(name, expression)`` pairs :func:`alpha158` compiles, Qlib's
    text unchanged."""
    return [tuple(pair) for pair in _phasor.alpha158_features(_alpha158_spec(config))]


def qlib_label(price: str = "close") -> FactorSet:
    """Qlib's training label ``LABEL0 = Ref($price, -2)/Ref($price, -1) - 1``:
    the return from the next bar to the one after, on ``close`` (Qlib's
    ``Alpha158`` / ``Alpha360``) or ``vwap`` (their ``vwap`` variants). It
    reads two bars ahead: a target, never a signal (``uses_future`` is
    true)."""
    return FactorSet(_phasor.qlib_label(price))


def alpha360() -> FactorSet:
    """Returns Qlib's default 360 compiled expressions."""
    return FactorSet(_phasor.alpha360())


def wq101(all: bool = False) -> FactorSet:
    """WorldQuant's 101 Formulaic Alphas (Kakushadze 2016), ``alpha001`` to
    ``alpha101``. By default the 82 that read only ``open``, ``high``,
    ``low``, ``close``, ``volume`` and ``vwap``; ``all=True`` adds the 19 that
    also read ``cap`` and the group labels ``sector``, ``industry``,
    ``subindustry``. ``wq101_formula(k)`` gives the paper's formula."""
    return FactorSet(_phasor.wq101(all))


def wq101_formula(k: int) -> str | None:
    """The paper's formula for alpha ``k`` (1 to 101)."""
    return _phasor.wq101_formula(k)


def _signals(value: Any, name: str) -> np.ndarray:
    """A 2-D, F-ordered, bool array."""
    array = np.asarray(value, dtype=bool)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got {array.ndim} dimensions")
    return np.asfortranarray(array)


def _market(value: Any, name: str) -> Any:
    """Market data beside the prices: ``None``, a number, or a matrix shaped like ``close``.

    A 1-D array is one column, as for ``close`` itself, not one value per column.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _matrix(value, name)


def _param(value: Any, name: str) -> Any:
    """A parameter, in whichever of the engine's four broadcast shapes it is.

    A number stays a number — the engine's ``FlexArray`` reads a scalar with no
    allocation at all, so converting one to an array here would spend memory to
    lose that. Arrays are converted; ``None`` means "the engine's default",
    which is where every default lives.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        # A NumPy scalar, or a 0-d array. Unwrapping it keeps the engine on its
        # allocation-free scalar path instead of building a one-element buffer.
        return float(array)
    if array.ndim == 2:
        return np.asfortranarray(array)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    raise ValueError(f"{name} must be a number, a 1-D array or a 2-D array")


def _flag(value: Any, name: str) -> Any:
    """A bool parameter: a scalar, a matrix, or ``None`` for the default."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        return bool(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a bool or a 2-D bool array")
    return np.asfortranarray(array)


def _per_bar(value: Any, close: Any, column_map: Any, name: str) -> Any:
    """A bool mask that may be one value per bar for every column (1-D)."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return _flag(value, name)
    array = np.asarray(value, dtype=bool)
    if array.ndim == 1:
        if column_map is not None:
            n_cols = len(column_map)
        else:
            n_cols = 1 if np.ndim(close) == 1 else np.shape(close)[1]
        array = np.repeat(array.reshape(-1, 1), n_cols, axis=1)
    return _flag(array, name)


class _TimeArgs:
    """Reads the time stops and order expiries as rows or as time."""

    def __init__(self, fmt: Any, index: Any, close: Any, column_map: Any):
        if fmt not in (None, "rows", "index"):
            raise ValueError(f"time_delta_format must be 'rows' or 'index'; got {fmt!r}")
        self.fmt = fmt
        self.index = index if index is not None else getattr(close, "index", None)
        self.n_rows = np.shape(close)[0]
        self.n_cols = 1 if np.ndim(close) == 1 else np.shape(close)[1]
        self.column_map = column_map

    def rows(self, value: Any, name: str, convert: Any, **kwargs: Any) -> Any:
        if value is None:
            return None
        as_time = self.fmt == "index" or (self.fmt is None and _timing.is_time(value))
        if not as_time:
            if _timing.is_time(value):
                raise ValueError(f"{name} is given as time; pass time_delta_format='index'")
            return _param(value, name)
        if self.index is None:
            raise ValueError(f"{name} as time needs the bars' times: pass index=")
        if len(self.index) != self.n_rows:
            raise ValueError(f"index has {len(self.index)} times for {self.n_rows} bars")
        m = convert(value, self.index, self.n_cols, **kwargs)
        if self.column_map is not None:
            m = np.asfortranarray(m[:, np.asarray(self.column_map, dtype=np.intp)])
        return m


_CASH_FREE_SIZES = {"amount", "value", "targetamount", "targetvalue"}


def _cash_needed(pf: Any) -> np.ndarray:
    """Each account's most cash out of pocket over the run, from its records:
    the lowest its cash ever went below where it started, or 0."""
    n_rows, n_cols = pf.n_rows, pf.n_cols
    flow = np.zeros((n_rows, n_cols))
    o = pf.orders()
    mult = o.get("multiplier", 1.0)
    sign = np.where(o["side"] == 0, -1.0, 1.0)
    np.add.at(flow, (o["idx"], o["col"]), sign * o["size"] * o["price"] * mult - o["fees"])
    c = pf.cash_records()
    np.add.at(flow, (c["idx"], c["col"]), c["amount"])
    lens = list(pf.group_lens) if pf.cash_sharing else [1] * n_cols
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.intp)
    accounts = np.add.reduceat(flow, starts, axis=1)
    lowest = np.minimum.accumulate(np.cumsum(accounts, axis=0), axis=0)[-1] if n_rows else 0.0
    return np.maximum(-lowest, 0.0)


def _auto_cash(run: Any, args: dict) -> Any:
    """``init_cash="auto"``/``"autoalign"``: see the entry points' docs."""
    mode = args["init_cash"]
    if mode not in ("auto", "autoalign"):
        raise ValueError(f"init_cash must be a number, an array, 'auto' or 'autoalign'; got {mode!r}")
    size_types = args.get("size_type", "amount")
    names = [size_types] if isinstance(size_types, str) else list(size_types)
    if not set(names) <= _CASH_FREE_SIZES:
        raise ValueError(
            f"init_cash={mode!r} needs sizes that do not depend on the cash "
            f"({', '.join(sorted(_CASH_FREE_SIZES))}); got size_type={size_types!r}"
        )
    leverage = args.get("leverage")
    if leverage is not None and np.any(np.asarray(leverage, dtype=np.float64) != 1.0):
        raise ValueError(f"init_cash={mode!r} needs cash accounts without leverage")
    account_type = args.get("account_type")
    if account_type is not None and set(np.atleast_1d(account_type).tolist()) != {"cash"}:
        raise ValueError(f"init_cash={mode!r} needs cash accounts without leverage")
    unlimited = run(**dict(args, init_cash=np.inf))
    need = _cash_needed(unlimited)
    if mode == "autoalign":
        need = np.full_like(need, need.max() if need.size else 0.0)
    pf = run(**dict(args, init_cash=need if need.size > 1 else float(need[0])))
    a, b = unlimited.orders(), pf.orders()
    same = all(np.array_equal(a[key], b[key]) for key in ("col", "idx", "size", "price", "side"))
    if not same:
        raise ValueError(
            f"init_cash={mode!r}: with the cash it needed the run fills differently from one "
            "with unlimited cash (a short's collateral, or a cap on cash, binds); pass init_cash"
        )
    return pf


def _call_seq(value: Any) -> Any:
    """``"default"``, ``"auto"``, or an explicit order: a ``(n_rows,
    n_cols)`` integer array whose row holds, for each group, the order its
    columns execute in as indices within the group."""
    if value is None or isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "iu" or (array.size and array.min() < 0):
        raise ValueError("call_seq must be 'default', 'auto' or a 2-D array of nonnegative integers")
    return np.asfortranarray(array.astype(np.uintp))


def _accumulate(value: Any) -> str:
    """The accumulation mode, accepting a boolean spelling.

    ``accumulate=True`` means "add and remove", and ``False`` means "ignore
    repeated signals". Those are two of the four modes, so both spellings work
    here.
    """
    if isinstance(value, (bool, np.bool_)):
        return "both" if value else "disabled"
    return value


def _group_lens(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError("group_lens must be a sequence of positive integers")
    lengths = list(value)
    if any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
        for n in lengths
    ):
        raise TypeError("group_lens must contain integers")
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("group_lens must contain positive lengths")
    return [int(n) for n in lengths]


def _account_type(value):
    if value is None or isinstance(value, str):
        return value
    try:
        names = list(value)
    except TypeError:
        names = None
    if names is None or not all(isinstance(name, str) for name in names):
        raise TypeError("account_type must be 'cash', 'margin', or one of them per column")
    return names


def _names(value, name):
    """A string parameter: one for every column, or a list of one per column."""
    if value is None or isinstance(value, str):
        return value
    try:
        names = list(value)
    except TypeError:
        names = None
    if names is None or not all(isinstance(n, str) for n in names):
        raise TypeError(f"{name} must be a string or one per column")
    return names


def _tiers(value: Any) -> dict:
    """``maintenance_margin_tiers`` as the engine's two tables, rows the
    tiers: ``margin_tier_notional`` and ``margin_tier_rate``."""
    if value is None:
        return {}
    tables = list(value)
    nested = tables and not np.isscalar(tables[0][0])
    columns = tables if nested else [tables]
    depth = max(len(c) for c in columns)
    bounds = np.full((depth, len(columns)), np.nan)
    rates = np.full((depth, len(columns)), np.nan)
    for j, tiers in enumerate(columns):
        for t, (bound, rate) in enumerate(tiers):
            bounds[t, j] = bound if np.isfinite(bound) else np.nan
            rates[t, j] = rate
    if not nested:
        return {"margin_tier_notional": np.ascontiguousarray(bounds[:, 0]),
                "margin_tier_rate": np.ascontiguousarray(rates[:, 0])}
    return {"margin_tier_notional": np.asfortranarray(bounds),
            "margin_tier_rate": np.asfortranarray(rates)}


def _margin_mode(value):
    if value is None or isinstance(value, str):
        return value
    try:
        names = list(value)
    except TypeError:
        names = None
    if names is None or not all(isinstance(name, str) for name in names):
        raise TypeError("margin_mode must be 'cross', 'isolated', or one of them per column")
    return names


def _column_map(value):
    if value is None:
        return None
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu" or a.size == 0 or np.any(a < 0):
        raise ValueError("column_map must be a nonempty one-dimensional array of nonnegative integers")
    return a.tolist()


def _multiplier(value: Any, column_map: Any, name: str = "multiplier") -> Any:
    """A per-column input (a contract multiplier, an initial position): a
    number, or one value per column of ``close``.

    With a ``column_map`` the values are per source column and are mapped
    here, as the engine wants one per simulated column.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a number or one value per column")
    if column_map is not None:
        array = array[np.asarray(column_map, dtype=np.intp)]
    return np.ascontiguousarray(array)


_PRICE_OPTIONS = {
    "open": ("open", 0),
    "close": ("close", 0),
    "nextopen": ("open", 1),
    "nextclose": ("close", 1),
    "nextvalidopen": ("open", None),
    "nextvalidclose": ("close", None),
}


def _from_ago(value: Any, close: Any, column_map: Any) -> Any:
    """``from_ago`` as the engine takes it: an integer, or a ``uintp``
    matrix one value per bar and column. A 1-D array is one value per bar for
    every column; a 2-D one is shaped like ``close`` and mapped through
    ``column_map``."""
    if value is None or isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if value is not None and value < 0:
            raise ValueError("from_ago must be 0 or more")
        return None if value is None else int(value)
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or (array.size and array.min() < 0):
        raise ValueError("from_ago must be nonnegative integers")
    if array.ndim == 0:
        return int(array)
    n_cols = 1 if np.ndim(close) == 1 else np.shape(close)[1]
    if array.ndim == 1:
        array = np.repeat(array.reshape(-1, 1), n_cols, axis=1)
    if array.ndim != 2:
        raise ValueError("from_ago must be an integer, a 1-D array or a 2-D array")
    if column_map is not None and array.shape[1] == n_cols:
        array = array[:, np.asarray(column_map, dtype=np.intp)]
    return np.asfortranarray(array.astype(np.uintp))


def _next_valid(close: Any, open: Any, column_map: Any) -> np.ndarray:
    """The ``from_ago`` of ``nextvalidopen`` / ``nextvalidclose``: every bar
    with a price executes the order decided on the last bar before it with a
    price, however many bars without one lie between. A bar without one
    executes nothing (its ``from_ago`` reaches before the first row)."""
    valid = np.isfinite(_matrix(close, "close"))
    if open is not None:
        valid &= np.isfinite(_matrix(open, "open"))
    n = valid.shape[0]
    rows = np.arange(n).reshape(-1, 1)
    last = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    before = np.vstack([np.full((1, valid.shape[1]), -1), last[:-1]])
    ago = np.where(valid & (before >= 0), rows - before, rows + 1)
    if column_map is not None:
        ago = ago[:, np.asarray(column_map, dtype=np.intp)]
    return np.asfortranarray(ago.astype(np.uintp))


def _price_option(price: Any, from_ago: Any, close: Any, open: Any, column_map: Any) -> tuple[Any, Any, Any]:
    """``price`` as the engine takes it: an array or number, or one of the
    bar's prices by name. ``nextopen`` and ``nextclose`` are the next bar's,
    which ``from_ago=1`` reads the order from the bar before;
    ``nextvalidopen`` and ``nextvalidclose`` skip bars without a price."""
    if not isinstance(price, str):
        return _param(price, "price"), None, _from_ago(from_ago, close, column_map)
    if price not in _PRICE_OPTIONS:
        raise ValueError(
            "price must be a number, an array, or 'open', 'close', 'nextopen', 'nextclose', "
            f"'nextvalidopen' or 'nextvalidclose'; got {price!r}"
        )
    price_type, ago = _PRICE_OPTIONS[price]
    if ago != 0 and from_ago is not None and not (np.ndim(from_ago) == 0 and from_ago == 0):
        raise ValueError(f"price={price!r} sets from_ago; do not pass from_ago with it")
    if ago is None:
        return None, price_type, _next_valid(close, open if price_type == "open" else None, column_map)
    return None, price_type, _from_ago(ago if ago else from_ago, close, column_map)


def from_signals(
    close,
    entries=None,
    exits=None,
    *,
    open=None,
    high=None,
    low=None,
    init_cash=None,
    init_position=None,
    init_price=None,
    allow_negative_price=False,
    size=None,
    size_type="amount",
    price=None,
    direction="longonly",
    fees=None,
    fixed_fees=None,
    slippage=None,
    min_size=None,
    max_size=None,
    size_granularity=None,
    leverage=None,
    leverage_mode="lazy",
    account_type=None,
    initial_margin=None,
    margin_basis=None,
    multiplier=None,
    from_ago=None,
    bid_price=None,
    ask_price=None,
    price_area_vio_mode=None,
    force_exit=None,
    cash_limit=None,
    reject_prob=None,
    seed=0,
    lock_cash=None,
    allow_partial=None,
    raise_reject=None,
    log=None,
    accumulate="disabled",
    sl_stop=None,
    sl_trail=None,
    tp_stop=None,
    stop_entry_price=None,
    stop_exit_price=None,
    upon_stop_exit=None,
    upon_stop_update=None,
    order_type=None,
    limit_delta=None,
    delta_format=None,
    limit_reverse=None,
    limit_order_price=None,
    limit_delay=None,
    limit_tif=None,
    limit_expiry=None,
    upon_adj_limit_conflict=None,
    upon_opp_limit_conflict=None,
    upon_adj_stop_conflict=None,
    upon_opp_stop_conflict=None,
    order_mode=False,
    tsl_stop=None,
    tsl_th=None,
    td_stop=None,
    dt_stop=None,
    stop_ladder=None,
    stop_order_type=None,
    stop_limit_delta=None,
    stop_exit_type=None,
    time_delta_format=None,
    index=None,
    val_price=None,
    cash_deposits=None,
    cash_dividends=None,
    cash_earnings=None,
    funding_rate=None,
    long_financing_rate=None,
    short_financing_rate=None,
    debt_rate=None,
    borrow_rate=None,
    mark_price=None,
    margin_mode=None,
    maintenance_margin=None,
    liquidation_fee=None,
    cash_maintenance_margin=None,
    maintenance_margin_tiers=None,
    liquidation_step=None,
    auto_deleverage=None,
    ffill_val_price=True,
    update_value=False,
    ann_factor=365.0,
    risk_free=0.0,
    trades="exit",
    incl_open=False,
    threads=1,
    cash_sharing=False,
    group_lens=None,
    call_seq="default",
    column_map=None,
    resume=None,
    row_offset=0,
    pre_group_func=None,
    signal_func=None,
    adjust_func=None,
    post_order_func=None,
    post_segment_func=None,
    risk=None,
    risk_stop=None,
    breakeven=None,
    max_positions=None,
    max_exposure=None,
) -> Portfolio:
    """Runs a signal-driven backtest and returns the portfolio.

    Parameters
    ----------
    close, entries, exits
        ``(n_bars, n_cols)``, or 1-D for a single column. These set the shape.
    column_map
        Logical run column -> source market column. Close and OHLC stay (T, S);
        signals and trading parameters use (T, len(column_map)).
    resume
        A portfolio to continue (``pf.update``): pass the whole history again,
        the bars it ran on and the new ones, with the same parameters, and only
        the new bars are simulated, from the portfolio's ``last_state``. The
        result is one portfolio over the whole history, equal to running it in
        one go, records in the same order. The closes of the bars it ran on
        must be the same; the starting cash is the portfolio's.
    row_offset
        With ``resume`` a ``SimState``, the low-level form a stepper uses: the
        arrays cover rows ``row_offset`` onwards of a longer history, and
        include the bars before the state's ``next_row`` that the run looks
        back at (the previous close for ``price=-inf``, ``from_ago`` bars).
        Returns a ``phasor.Segment``: the new rows' records, on the history's
        rows, and its ``last_state``. Not a portfolio: it starts mid-run.
    open, high, low
        The rest of the bar. Needed for stops and for an intrabar liquidation
        test, and **required** for stops:
        a stop level is tested against the bar's range, so without ``high`` and
        ``low`` no stop can fire and the run would silently report a stopless
        strategy. Passing ``sl_stop`` or ``tp_stop`` without them is an error
        rather than a surprise.
    init_cash, size, fees, fixed_fees, slippage, min_size, max_size, size_granularity
        A number, one value per column, or a full matrix. ``None`` takes the
        engine's own default, which is the only place a default is written down.
    size_type
        ``amount``, ``value``, ``percent``, or a target: ``targetamount``,
        ``targetvalue``, ``targetpercent``. With a target an entry moves the
        position to ``size`` (``-size`` for a short) from flat, from the other
        side, or, where ``accumulate`` lets it add, from the same side; an
        exit closes the position. Limit orders, bid and ask, and conflicts
        read a target order's side off the target's sign.
    direction
        ``longonly``, ``shortonly`` or ``both``.
    accumulate
        ``disabled``, ``both``, ``addonly`` or ``removeonly`` — whether a
        repeated signal adds to or trims an open position rather than being
        ignored. ``True``/``False`` also work: ``True`` is ``both``, ``False`` is ``disabled``.
    ann_factor
        Periods per year, for the annualized metrics. 365 for daily crypto,
        252 for daily equities, 365*24 for hourly.
    trades
        Which trade view the statistics are scored over: ``exit``, ``entry`` or
        ``positions``. They give different trade counts and different per-trade
        PnL from identical fills, and none of the three is wrong — a win rate
        over entry trades answers a different question than one over exits.
    log
        ``True`` to record why every order ended the way it did, or a
        ``(n_bars, n_cols)`` mask to log one column or one row. Off by default:
        a log row is 46 fields against an order record's 7, written for every
        order *considered* rather than every one filled. Read them back with
        ``pf.logs()``, ``pf.log_reasons()`` and ``pf.reason_counts(col)``.
    raise_reject
        Turn a rejected order into an exception instead of a recorded refusal.
        Useful under ``log=True`` to stop at the first one.
    sl_stop, sl_trail, tp_stop
        Stop-loss and take-profit distances as fractions of the stop's anchor
        price (NaN disables one), and whether the stop-loss trails the best
        price reached. Need ``high`` and ``low``.
    stop_entry_price
        The anchor a stop is measured from when its position opens or is
        re-armed: ``close`` (the default), ``fillprice`` (the fill, slippage
        included), ``price`` (the requested order price) or ``valprice``.
    stop_exit_price
        What a triggered stop fills at: ``stoplimit`` (the default — the stop
        level, no slippage), ``stopmarket`` (the stop level plus slippage),
        ``price`` (the ``price`` parameter) or ``close``.
    upon_stop_exit
        What a triggered stop does: ``close`` (the default), ``reverse`` (close
        and open the opposite side — opening it needs ``direction="both"``),
        or ``closereduce``/``reversereduce``, which leave ``accumulate`` in
        force instead of switching it off for that order.
    upon_stop_update
        When a position is added to, whether the new ``sl_stop``/``tp_stop``
        re-anchors the live stop at the new price: ``override`` (the default;
        a NaN keeps the old stop), ``overridenan`` (a NaN clears it) or
        ``keep``. Averaging down with the original stop and averaging down
        with a moved one are different strategies.
    order_type
        ``"market"`` (the default) executes a signal on its bar; ``"limit"``
        rests it as a limit order until the market reaches it. A string, or
        one per column. Stop exits are always market orders. A limit order
        pays no slippage; it fills at once if the order price already reaches
        it, and is otherwise tested from the next bar (from the rest of its
        own bar when placed at the open, ``price=-np.inf``). While it rests,
        a signal on the same side is ignored and one on the other side
        cancels it, as does a stop or a liquidation filling in its column.
    limit_delta, delta_format, limit_reverse
        Where it rests: ``limit_delta`` below the order price for a buy and
        above for a sell, as a fraction (``delta_format="percent"``, the
        default), a percentage (``"percent100"``), an amount of price
        (``"absolute"``) or the limit price itself (``"target"``); NaN is at
        the order price. ``limit_reverse`` turns both round, so a buy fills at
        or above its price: a breakout entry. ``delta_format`` reads the stop
        distances too (``sl_stop``, ``tsl_stop``, ``tsl_th``, ``tp_stop``):
        ``"absolute"`` makes them amounts of price from the entry,
        ``"target"`` the stop levels themselves.
    limit_order_price
        What it fills at once reached: ``"limit"`` (the default; the open
        instead when a bar opens beyond the limit), ``"hardlimit"`` (the
        limit even then) or ``"close"``.
    limit_delay, limit_tif, limit_expiry
        ``limit_delay`` tests it from the next bar even when the price already
        reaches it. ``limit_tif`` is how many bars after the one it was placed
        on it may fill, and ``limit_expiry`` the row it is cancelled at; NaN
        is no limit. Either may be given as time instead: see
        ``time_delta_format``.
    upon_adj_limit_conflict, upon_opp_limit_conflict
        What a new signal does to a resting limit order on the same side
        (``adj``) or the other (``opp``): ``"keepignore"`` (the ``adj``
        default) keeps it and ignores the signal, ``"cancelexecute"`` (the
        ``opp`` default) cancels it and executes the signal, and
        ``"keepexecute"`` and ``"cancelignore"`` the other two. A kept order
        keeps the size it was placed with; a new limit order takes its place.
    order_mode
        ``True`` reads ``size`` as the orders, as ``from_orders`` does, instead
        of the signals: each bar's ``size`` (NaN or 0: none) in ``size_type``,
        on ``direction``'s sides (set ``direction="both"`` to sell short).
        ``entries`` and ``exits`` may be left out. Stops, limit orders and
        the price options apply as to a signal's order.
    tsl_stop, tsl_th
        A trailing stop ``tsl_stop`` below the best price since the entry (above
        it for a short), live once the price has first moved ``tsl_th`` in the
        position's favour (NaN or 0: at once).
    td_stop, dt_stop
        Time stops: exit at the open ``td_stop`` bars after the entry bar, or
        at the open of row ``dt_stop``; or, as time, see
        ``time_delta_format``.
    time_delta_format, index
        How ``td_stop``, ``dt_stop``, ``limit_tif`` and ``limit_expiry`` are
        read. ``"rows"`` counts bars and rows, as above. ``"index"`` reads
        them against the bars' times, ``index`` (a ``DatetimeIndex`` or a
        ``datetime64`` array; ``close.index`` when ``close`` has one): a
        duration (``"7D"``, ``"90min"``, a ``timedelta``) for ``td_stop`` and
        ``limit_tif`` — exit at the first bar that long after the entry bar's
        time, fill on the bars before that long has passed — and for
        ``dt_stop`` and ``limit_expiry`` a moment (``"2024-03-01"``), the end
        of the period a bar is in (``"D"``, ``"W"``, ``"ME"``) or the next
        time of day (``"18:00"``, in the index's time zone). ``None`` (the
        default) is ``"index"`` for such values and ``"rows"`` for numbers.
        Each may be one value for every column or a list of one per column.
    stop_ladder
        ``"uniform"``, ``"weighted"``, ``"adaptuniform"`` or
        ``"adaptweighted"`` makes ``sl_stop``, ``tsl_stop`` and ``tp_stop``
        ladders: a 1-D array of increasing steps for every column, or
        ``(n_steps, n_cols)``, NaN after the last. Each step closes an equal
        share of the position the ladder was armed on (``uniform``), or a
        share proportional to its distance from the step before
        (``weighted``); the ``adapt`` modes take the share of what is held
        when the step fires instead (an equal share over the steps left, or
        the step's distance over the distance left). The last step closes the
        rest.
    stop_order_type, stop_limit_delta
        ``"limit"`` turns a fired stop into a limit order at its level (a
        time stop's: the open), ``stop_limit_delta`` (in ``delta_format``)
        worse: a stop-limit order, which a gap past the limit leaves resting.
        A string or one per column.
    stop_exit_type
        An alias of ``upon_stop_exit``; pass one or the other.
    from_ago, bid_price, ask_price, price_area_vio_mode
        ``price`` may also name one of the bar's prices: ``"open"``,
        ``"close"``, or ``"nextopen"``/``"nextclose"``, the next bar's, which
        reads the order decided on the bar before (``from_ago=1``): a
        decision at one close executed at the following open uses nothing it
        could not have known. ``from_ago=n`` reads the order (its signals or
        size, size type and an explicit price) ``n`` bars back; a named price
        is still the executing bar's. ``from_ago`` may also be one value per
        bar (1-D) or shaped like ``close``. ``"nextvalidopen"`` and
        ``"nextvalidclose"`` skip bars without a price (``NaN`` close, or
        open): the order decided on the last bar with one executes on the
        next bar with one. ``bid_price`` and ``ask_price``, shaped
        like ``close``, are what a sell and a buy execute at where given.
        ``price_area_vio_mode`` decides what an order price outside the bar's
        low–high range does: ``"ignore"`` (the default), ``"cap"`` it to the
        range, or ``"error"``.
    force_exit
        A mask of bars that close the position at the order price in place of
        anything else on them: the end of a trading session
        (``phasor.sessions.session_end``). In ``from_signals`` a stop that
        fires inside such a bar fills first, and a resting limit order is
        cancelled.
    upon_adj_stop_conflict, upon_opp_stop_conflict
        The same choice between a stop that fires and the bar's own signal.
        The default ``"keepignore"`` lets the stop replace the signal.
        ``"keepexecute"`` fills the stop, then the signal re-sized against
        what the stop left, on the same bar (a stop-out and a reversal);
        ``"cancelexecute"`` disarms the stop and executes the signal instead.
    cash_sharing, group_lens
        Share capital across adjacent columns. None groups all columns when
        sharing is enabled; explicit positive lengths must sum to n_cols.
        Shared init_cash is a scalar or one value per group.
    init_position, init_price
        Units held before the first bar, a number or one per column of
        ``close`` (negative: short), and the price they were opened at (NaN:
        the first close). The position is recorded as a fill on the first
        bar at that price with no fees, so it is part of the trades. A cash
        column paid for it before the run: an ``"initialposition"`` cash
        record of its cost balances the fill, the cash stays ``init_cash``,
        and the account starts worth ``init_cash`` plus the position, which
        returns count as capital brought in. A margin column holds it as an
        open trade, its margin out of free cash. No stop is armed for it.
    init_cash="auto", init_cash="autoalign"
        The cash the strategy needed: the run is made with unlimited cash,
        and each account (``"autoalign"``: every account) is then given the
        most it was ever out of pocket, net of deposits, and the run made
        again. Only for sizes that do not depend on the cash — ``amount``,
        ``value``, ``targetamount``, ``targetvalue`` — and cash accounts
        without leverage; an error if the second run fills differently.
    allow_negative_price
        Treats a price of 0 or below as a price, as a futures contract's can
        be (WTI in April 2020), rather than refusing it. Such orders must be
        sized in units (``amount``, ``targetamount``). A cash column may buy
        and sell what it holds; a margin column trades either side. Fees,
        slippage and margin are taken on the price's size, slippage always
        against the trade, and a trade's return is its profit over the size
        of its entry value. Percent stop distances, holding charges and
        liquidation assume prices above 0.
    call_seq
        "default" uses column order; "auto" stably sorts approximate order
        values, selling before buying. Neither mode allocates equal weights.
        An integer array shaped ``(n_bars, n_cols)`` gives the order itself:
        on each bar, each group's columns hold the indices within the group
        of the columns to execute first, second, and so on.
    val_price, ffill_val_price, update_value
        Valuation price (default: execution price), whether missing valuation
        prices retain the previous value, and whether each fill updates the
        account value before the next order. These do not shift signals.
    threads
        Splits independent account groups; symbols sharing cash execute sequentially.
    cash_deposits
        Cash added (positive) or withdrawn (negative) at each bar's open,
        before orders are sized, so a target-sized order on the same bar
        invests it. A number, one value per bar (1-D), or a matrix: one column
        per account — ``(n_bars, n_cols)`` unshared, ``(n_bars, n_groups)``
        with ``cash_sharing``. A withdrawal is capped at the free cash.
        Deposits are taken out of returns: each bar's return is measured from
        the previous value plus that bar's deposit.
    cash_limit
        The most cash one buy may spend, fees included. Applied before
        ``percent`` takes its share. Sells and shorts are not affected.
    leverage, leverage_mode
        How many times its own cash a buy may be worth: 1 or more, as a
        number, one value per column or a matrix. ``lazy`` (the default)
        spends own cash first and borrows only what it cannot cover;
        ``eager`` borrows ``1 - 1/leverage`` of every buy. Fees are paid from
        own cash, and ``percent`` sizes against the levered buying power. A
        short ties up its proceeds plus ``value / leverage`` (twice its value
        at 1). A buy with no own cash left is refused as
        ``NoCashLeverage``. Under leverage ``pf.cash()`` is own cash less what
        the longs borrowed and can go negative, ``pf.value()`` stays the
        equity, and ``pf.orders()`` gains ``debt`` and ``locked_cash``.
    account_type, initial_margin, margin_basis
        ``cash`` (the default) holds what it buys; ``margin`` trades like a
        futures or perpetual contract on the account's cash. A string for
        every column or one per column, so a spot leg and a perpetual leg can
        share one account with ``cash_sharing``. Opening either side of a
        margin position posts ``initial_margin`` from free cash: a fraction of
        the notional under ``margin_basis="notional"`` (the default; ``None``
        initial margin means ``1 / leverage``), or cash per unit under
        ``"unit"``; ``margin_basis`` is a string or one per column. Every bar the
        position's profit since the last bar is settled into cash, so a loss
        shrinks what new orders may post and cash can go below zero. Closing
        realizes against the average entry and releases the margin.
        ``leverage_mode`` and ``lock_cash`` do not apply to margin columns.
        ``pf.value()`` is the equity, ``pf.free_cash()`` the equity less the
        margin posted, and ``pf.orders()`` gains ``debt`` (the cost basis),
        ``locked_cash`` (the margin) and ``margin``.
    multiplier
        What one unit of position is worth per unit of price: 50 for an
        E-mini S&P 500 future, 1 (the default) for a share. A number, or one
        value per column (per source column with ``column_map``); it cannot
        change within a column. A fill moves ``size * price * multiplier`` of
        cash, and fees, margin, dividends, holding charges and the
        liquidation test are all on that notional. Sizes, ``min_size``,
        ``size_granularity`` and a ``"unit"`` margin stay in contracts, and
        prices, stops and ``pf.orders()["price"]`` stay prices. A run with a
        multiplier other than 1 adds ``multiplier`` to ``pf.orders()``, as a
        levered run does; ``pf.trades()`` always has it, with ``pnl``
        multiplied and ``ret`` not.
    reject_prob, seed
        Probability that an order is rejected at random, for testing how a
        strategy degrades when fills fail. Each draw depends only on ``seed``,
        the bar and the column, so a run is reproducible on any thread count.
    cash_dividends
        Dividend per share, shaped like ``close`` (``column_map`` applies). Paid
        at each bar's open on the position held before that bar's orders; a
        short position pays it. Part of the return.
    cash_earnings
        Cash credited (positive) or charged (negative) at each bar's close, per
        column: interest, fees, a funding payment computed in advance. A
        number, one value per bar (1-D), or ``(n_bars, n_cols)``. Part of the
        return. A charge is capped at the free cash. ``pf.cash_records()``
        lists what was applied.
    funding_rate
        A perpetual's funding rate per bar, shaped like ``close``
        (``column_map`` applies), ``NaN`` or 0 on bars that do not settle. At
        the bar's close, after its orders, a position pays ``position * price
        * rate``: a positive rate is paid by longs and received by shorts. Put
        each rate on the bar whose close is its funding time.
    long_financing_rate, short_financing_rate
        Overnight financing per bar for each side, shaped like ``close``: a
        position pays ``abs(position) * price * rate`` at the bar's close. A
        positive rate is a cost to either side, a negative one is received.
    debt_rate
        Interest per bar on what a cash column's long borrowed under
        ``leverage``, charged at the bar's close. A number, one value per bar
        (1-D) or ``(n_bars, n_cols)``.
    borrow_rate
        The fee per bar for the shares a cash column's short borrowed, shaped
        like ``close``: ``abs(position) * price * rate`` at the bar's close.
    mark_price
        The ``price`` of the charges above, shaped like ``close``. ``None``
        uses the close.

        Every rate is per bar, for the time until the next bar;
        ``phasor.rates.per_bar`` turns an annual rate into one from the bars'
        timestamps. A charge is owed whatever the balance, so unlike a
        ``cash_earnings`` charge it is not capped: any of these runs the
        accounts as ones that may borrow, cash can go below zero, after which
        only closing is allowed, and ``pf.orders()`` gains ``debt``,
        ``locked_cash`` and ``margin``. Margin columns pay funding and
        financing but no interest or borrow fee. ``pf.cash_flows("funding")``,
        ``"financing"``, ``"interest"`` and ``"borrowfee"`` give what each
        cost per bar and column; all are part of the return.
    margin_mode, maintenance_margin, liquidation_fee
        Liquidation of margin columns. ``maintenance_margin`` (in
        ``margin_basis``: a fraction of the notional below 1, or cash per unit;
        ``None`` or 0 carries none and never liquidates) is tested on every bar
        before its orders, along a path from the bar's open to its adverse
        extreme — the low for a long, the high for a short — or to
        ``mark_price`` where one is given. An account whose equity falls
        below the maintenance margin is closed where its path first crosses
        it, or at the open when it gaps through, paying ``liquidation_fee`` of
        the notional instead of its fees. ``margin_mode`` is ``"cross"`` (the
        default: the account's equity carries every cross position and they
        close together; cash columns count as collateral and are never closed)
        or ``"isolated"`` (the position is not settled into the account's
        cash, is tested on its own margin alone and can lose no more than it
        posted), a string for every column or one per column. A loss a flat
        cross account or an isolated position cannot pay is written off, as an
        exchange's insurance fund takes it: ``pf.cash_flows("liquidation")``.
        Fills that were liquidations are flagged in ``pf.orders()`` and
        ``pf.logs()`` (``liquidation``). Without ``high`` and ``low`` the test
        uses the close. Each leg of a hedge reaches its own extreme, so a
        hedged account wants a mark price, if only the close. An isolated
        column pays its funding and financing out of its own margin.
    cash_maintenance_margin
        The maintenance margin of cash columns that borrow (margin lending:
        ``leverage`` above 1, or shorting), a fraction of the position's
        notional below 1. Below it the account's cash columns that carry one
        are liquidated with its margin columns, as a broker's margin call
        does. ``None`` (the default) keeps cash columns as collateral that is
        never closed.
    maintenance_margin_tiers
        An exchange's tier table in place of ``maintenance_margin``: a list of
        ``(notional_up_to, rate)`` pairs, increasing, the last bound ignored
        (``[(50_000, 0.004), (250_000, 0.005), (float("inf"), 0.01)]``), for
        every margin column, or one such list per column. The requirement is
        ``rate × notional − amount`` for the tier the notional falls in, the
        amount keeping it continuous at the bounds.
    liquidation_step
        The fraction of a position each breach closes, above 0 and at most 1;
        ``None`` (the default) closes it all. Below 1 the test goes on along
        the rest of the bar after each partial close, closing the same
        fraction again at each further breach.
    auto_deleverage
        A mask of bars on which the exchange auto-deleverages: a margin
        column's position is closed at ``mark_price`` (the close without one),
        with no fee, flagged as a liquidation. 1-D is one value per bar for
        every column. Which bars that hits is the caller's to say: it is the
        exchange's insurance fund, not the account, that decides.

    pre_group_func, signal_func, adjust_func, post_order_func, post_segment_func
        Python functions called inside the loop (see ``from_order_func`` for
        the cost). ``pre_group_func(group, from_col, to_col)`` returns the
        group's state, passed last to every other call and returned in
        ``pf.in_outputs``. ``signal_func(ctx, k, state)`` decides column
        ``k`` (its place in the group) on this bar: ``None`` keeps the
        arrays', a tuple ``(long_entry, long_exit, short_entry,
        short_exit)``, or a dict of those and ``size``/``size_type``.
        ``adjust_func(ctx, k, stops, state)`` gets the column's stops as a
        dict before they are tested and returns ``None`` or the fields to
        change (``sl_stop``, ``sl_price``, ``tp_stop``, ``tsl_stop``...).
        ``post_order_func(ctx, k, result, state)`` and
        ``post_segment_func(ctx, state)`` return ``None``, or ``"stop"`` to
        stop trading the group. ``ctx`` is a ``phasor.Context``: ``row``,
        ``group``, ``cash``, ``free_cash``, ``value`` and per-column
        ``position``, ``debt``, ``val_price``.

    risk, risk_stop, breakeven, max_positions, max_exposure
        Built-in rules, run in Rust without Python callbacks (and not
        combined with them); each is off until set. ``risk`` with
        ``risk_stop``: an order that opens or reverses a position takes the
        size whose loss at ``risk_stop`` (a fraction of the bar's close)
        against it is ``risk`` of the account's value. ``breakeven``: once
        the last close is that fraction past the entry in the position's
        favour, the stop-loss moves to the entry price (never loosening a
        tighter one). ``max_positions``: a signal that would open a new
        position is dropped while the account holds that many (counting
        ones opened earlier on the bar); ``max_exposure``: likewise while
        its gross exposure is at least that multiple of its value. Exits are
        never dropped.

    Returns
    -------
    Portfolio
        The run's fills, plus everything derivable from them on demand.
    """
    _resume(resume, init_cash)
    if isinstance(init_cash, str):
        return _auto_cash(from_signals, dict(locals()))
    price, price_type, from_ago = _price_option(price, from_ago, close, open, column_map)
    if stop_exit_type is not None:
        if upon_stop_exit is not None:
            raise ValueError("stop_exit_type is upon_stop_exit's other name; pass one of them")
        upon_stop_exit = stop_exit_type
    when = _TimeArgs(time_delta_format, index, close, column_map)
    return _phasor.from_signals(
        _matrix(close, "close"),
        False if entries is None else _signals(entries, "entries"),
        False if exits is None else _signals(exits, "exits"),
        open=None if open is None else _matrix(open, "open"),
        high=None if high is None else _matrix(high, "high"),
        low=None if low is None else _matrix(low, "low"),
        init_cash=_param(init_cash, "init_cash"),
        size=_param(size, "size"),
        size_type=size_type,
        price=price,
        direction=direction,
        fees=_param(fees, "fees"),
        fixed_fees=_param(fixed_fees, "fixed_fees"),
        slippage=_param(slippage, "slippage"),
        min_size=_param(min_size, "min_size"),
        max_size=_param(max_size, "max_size"),
        size_granularity=_param(size_granularity, "size_granularity"),
        leverage=_param(leverage, "leverage"),
        leverage_mode=leverage_mode,
        account_type=_account_type(account_type),
        initial_margin=_param(initial_margin, "initial_margin"),
        margin_basis=_names(margin_basis, "margin_basis"),
        multiplier=_multiplier(multiplier, column_map),
        init_position=_multiplier(init_position, column_map, "init_position"),
        init_price=_multiplier(init_price, column_map, "init_price"),
        allow_negative_price=bool(allow_negative_price),
        price_type=price_type,
        from_ago=from_ago,
        bid_price=_market(bid_price, "bid_price"),
        ask_price=_market(ask_price, "ask_price"),
        price_area_vio_mode=price_area_vio_mode,
        force_exit=_per_bar(force_exit, close, column_map, "force_exit"),
        cash_limit=_param(cash_limit, "cash_limit"),
        reject_prob=_param(reject_prob, "reject_prob"),
        seed=seed,
        lock_cash=_flag(lock_cash, "lock_cash"),
        allow_partial=_flag(allow_partial, "allow_partial"),
        raise_reject=_flag(raise_reject, "raise_reject"),
        log=_flag(log, "log"),
        accumulate=_accumulate(accumulate),
        sl_stop=_param(sl_stop, "sl_stop"),
        sl_trail=_flag(sl_trail, "sl_trail"),
        tp_stop=_param(tp_stop, "tp_stop"),
        stop_entry_price=stop_entry_price,
        stop_exit_price=stop_exit_price,
        upon_stop_exit=upon_stop_exit,
        upon_stop_update=upon_stop_update,
        order_type=order_type,
        limit_delta=_param(limit_delta, "limit_delta"),
        delta_format=delta_format,
        limit_reverse=_flag(limit_reverse, "limit_reverse"),
        limit_order_price=limit_order_price,
        limit_delay=_flag(limit_delay, "limit_delay"),
        limit_tif=when.rows(limit_tif, "limit_tif", _timing.duration_rows, inclusive=True),
        limit_expiry=when.rows(limit_expiry, "limit_expiry", _timing.deadline_rows),
        upon_adj_limit_conflict=upon_adj_limit_conflict,
        upon_opp_limit_conflict=upon_opp_limit_conflict,
        upon_adj_stop_conflict=upon_adj_stop_conflict,
        upon_opp_stop_conflict=upon_opp_stop_conflict,
        order_mode=bool(order_mode),
        tsl_stop=_param(tsl_stop, "tsl_stop"),
        tsl_th=_param(tsl_th, "tsl_th"),
        td_stop=when.rows(td_stop, "td_stop", _timing.duration_rows, inclusive=False),
        dt_stop=when.rows(dt_stop, "dt_stop", _timing.deadline_rows),
        stop_ladder=stop_ladder,
        stop_order_type=stop_order_type,
        stop_limit_delta=_param(stop_limit_delta, "stop_limit_delta"),
        val_price=_param(val_price, "val_price"),
        cash_deposits=_param(cash_deposits, "cash_deposits"),
        cash_dividends=_market(cash_dividends, "cash_dividends"),
        cash_earnings=_param(cash_earnings, "cash_earnings"),
        funding_rate=_market(funding_rate, "funding_rate"),
        long_financing_rate=_market(long_financing_rate, "long_financing_rate"),
        short_financing_rate=_market(short_financing_rate, "short_financing_rate"),
        debt_rate=_param(debt_rate, "debt_rate"),
        borrow_rate=_market(borrow_rate, "borrow_rate"),
        mark_price=_market(mark_price, "mark_price"),
        margin_mode=_margin_mode(margin_mode),
        maintenance_margin=_param(maintenance_margin, "maintenance_margin"),
        liquidation_fee=_param(liquidation_fee, "liquidation_fee"),
        cash_maintenance_margin=_param(cash_maintenance_margin, "cash_maintenance_margin"),
        **_tiers(maintenance_margin_tiers),
        liquidation_step=_param(liquidation_step, "liquidation_step"),
        auto_deleverage=_per_bar(auto_deleverage, close, column_map, "auto_deleverage"),
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        ann_factor=ann_factor,
        risk_free=risk_free,
        trades=trades,
        incl_open=incl_open,
        threads=threads,
        cash_sharing=cash_sharing,
        group_lens=_group_lens(group_lens),
        call_seq=_call_seq(call_seq),
        column_map=_column_map(column_map),
        resume=_resume(resume, init_cash),
        row_offset=int(row_offset),
        callbacks=_callbacks_of(
            pre_group=pre_group_func,
            signal=signal_func,
            adjust=adjust_func,
            post_order=post_order_func,
            post_segment=post_segment_func,
        ),
        builtin=_builtin(risk, risk_stop, breakeven, max_positions, max_exposure),
    )


def _builtin(risk, risk_stop, breakeven, max_positions, max_exposure):
    rules = {"risk": risk, "risk_stop": risk_stop, "breakeven": breakeven,
             "max_positions": max_positions, "max_exposure": max_exposure}
    rules = {k: v for k, v in rules.items() if v is not None}
    return rules or None


def _resume(resume, init_cash):
    if resume is None:
        return None
    if not isinstance(resume, (Portfolio, SimState)):
        raise TypeError(f"resume must be a Portfolio or a SimState; got {type(resume).__name__}")
    if isinstance(init_cash, str):
        raise ValueError("init_cash='auto' sizes a run's start; a resumed run continues one")
    return resume


_OBJECT = b"PHOB\x01"


def save(obj, path):
    """Writes ``obj`` to ``path``.

    A portfolio is written as its versioned bytes: its records, logs, the
    inputs its analysis reads and its state, so ``load(path)`` gives a
    portfolio that analyses, and continues with ``resume=``, as it does.
    Callback states (``in_outputs``) are not kept. Any other research object
    (``params.ParamResults``, an ``IndicatorResult``, a dict of arrays, a
    list of portfolios) is pickled, portfolios inside it as their bytes.
    Load pickles only from sources you trust."""
    import pickle

    with open(path, "wb") as f:
        if isinstance(obj, Portfolio):
            f.write(obj.to_bytes())
        else:
            f.write(_OBJECT)
            pickle.dump(_Persist(obj), f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    """What ``save`` wrote."""
    import pickle

    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(_OBJECT):
        return pickle.loads(data[len(_OBJECT):]).obj
    return Portfolio.from_bytes(data)


class _Persist:
    """Pickles portfolios anywhere inside an object as their bytes."""

    def __init__(self, obj):
        self.obj = obj

    def __reduce__(self):
        return (_Persist._restore, (_Persist._pack(self.obj),))

    @staticmethod
    def _pack(obj):
        if isinstance(obj, Portfolio):
            return ("__phasor_portfolio__", obj.to_bytes())
        if isinstance(obj, dict):
            return {k: _Persist._pack(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_Persist._pack(v) for v in obj)
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            clone = object.__new__(type(obj))
            clone.__dict__.update({k: _Persist._pack(v) for k, v in vars(obj).items()})
            return clone
        return obj

    @staticmethod
    def _unpack(obj):
        if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__phasor_portfolio__":
            return Portfolio.from_bytes(obj[1])
        if isinstance(obj, dict):
            return {k: _Persist._unpack(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_Persist._unpack(v) for v in obj)
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            obj.__dict__.update({k: _Persist._unpack(v) for k, v in vars(obj).items()})
        return obj

    @staticmethod
    def _restore(packed):
        return _Persist(_Persist._unpack(packed))


def _callbacks_of(**funcs):
    funcs = {name: f for name, f in funcs.items() if f is not None}
    for name, f in funcs.items():
        if not callable(f):
            raise TypeError(f"{name}_func must be callable; got {type(f).__name__}")
    return funcs or None


def from_order_func(
    close,
    order_func,
    *,
    flexible=False,
    row_wise=False,
    pre_group_func=None,
    pre_segment_func=None,
    post_order_func=None,
    post_segment_func=None,
    size=np.nan,
    **kwargs,
) -> Portfolio:
    """``from_orders`` with the orders decided by Python functions, bar by bar.

    Every other argument is ``from_orders``': prices, fees, accounts, cash
    flows, grouping and the rest. ``size`` is each order before
    ``order_func`` sees it, ``NaN`` (no order) by default.

    Each call takes the GIL, so this is orders of magnitude slower than an
    array-driven run: for trying a rule out and for debugging. A rule worth
    running at scale belongs in Rust (``phasor_sim::callbacks``).

    order_func
        ``order_func(ctx, k, state)`` for column ``k`` (its place in the
        group) on bar ``ctx.row``, in the group's call sequence. Returns
        ``None`` (keep the arrays' order), a size, or a dict of order fields:
        ``size``, ``price`` (``inf`` is the close), ``size_type``,
        ``direction``, ``fees``, ``slippage``, ``leverage``... With
        ``flexible=True`` it is ``order_func(ctx, call, state)`` instead,
        called until it returns ``None``, each call returning ``(k,
        order)``: any number of orders per bar, for any column, in any
        sequence.
    row_wise
        Run every group's bar before any group's next bar, on one thread, so
        that functions sharing state across groups see the bars in order.
        The records are the same as without when they do not.
    pre_group_func
        ``pre_group_func(group, from_col, to_col)`` returns the group's state
        (``None`` without it), passed last to every other call and returned
        in ``pf.in_outputs``, in group order. Mutate it in place.
    pre_segment_func
        ``pre_segment_func(ctx, state)`` before the bar's orders: ``"skip"``
        places none, ``"stop"`` stops trading the group.
    post_order_func
        ``post_order_func(ctx, k, result, state)`` after each order, filled or
        not; ``result`` is a dict (``filled``, ``size``, ``price``, ``fees``,
        ``side``, ``status``, ``status_info``). ``"stop"`` stops the group.
    post_segment_func
        ``post_segment_func(ctx, state)`` at the bar's end, after earnings and
        holding charges. ``"stop"`` stops the group.

    ``ctx`` is a ``phasor.Context``: ``row``, ``group``, ``from_col``,
    ``to_col``, ``cash``, ``free_cash``, ``value``, and per-column lists
    ``position``, ``debt``, ``locked_cash``, ``val_price``, ``multiplier``.
    A stopped group places no more orders, but its open positions still
    earn, pay holding charges and can be liquidated.
    """
    for name, f in [("order_func", order_func)]:
        if not callable(f):
            raise TypeError(f"{name} must be callable; got {type(f).__name__}")
    callbacks = _callbacks_of(
        order=order_func,
        pre_group=pre_group_func,
        pre_segment=pre_segment_func,
        post_order=post_order_func,
        post_segment=post_segment_func,
    )
    return from_orders(
        close,
        size,
        _callbacks=callbacks,
        _flexible=bool(flexible),
        _row_wise=bool(row_wise),
        **kwargs,
    )


def from_orders(
    close,
    size=None,
    *,
    open=None,
    high=None,
    low=None,
    size_type="amount",
    price=None,
    direction="both",
    init_cash=None,
    init_position=None,
    init_price=None,
    allow_negative_price=False,
    fees=None,
    fixed_fees=None,
    slippage=None,
    min_size=None,
    max_size=None,
    size_granularity=None,
    leverage=None,
    leverage_mode="lazy",
    account_type=None,
    initial_margin=None,
    margin_basis=None,
    multiplier=None,
    from_ago=None,
    bid_price=None,
    ask_price=None,
    price_area_vio_mode=None,
    force_exit=None,
    cash_limit=None,
    reject_prob=None,
    seed=0,
    lock_cash=None,
    allow_partial=None,
    raise_reject=None,
    log=None,
    val_price=None,
    cash_deposits=None,
    cash_dividends=None,
    cash_earnings=None,
    funding_rate=None,
    long_financing_rate=None,
    short_financing_rate=None,
    debt_rate=None,
    borrow_rate=None,
    mark_price=None,
    margin_mode=None,
    maintenance_margin=None,
    liquidation_fee=None,
    cash_maintenance_margin=None,
    maintenance_margin_tiers=None,
    liquidation_step=None,
    auto_deleverage=None,
    ffill_val_price=True,
    update_value=False,
    ann_factor=365.0,
    risk_free=0.0,
    trades="exit",
    incl_open=False,
    threads=1,
    cash_sharing=False,
    group_lens=None,
    call_seq="default",
    column_map=None,
    resume=None,
    row_offset=0,
    _callbacks=None,
    _flexible=False,
    _row_wise=False,
) -> Portfolio:
    """Runs a backtest from one order request per element.

    The other shape of a vectorized run. Where ``from_signals`` resolves
    entries and exits into sizes, here ``size`` *is* the request, element by
    element: a rebalance is a matrix of target weights, a schedule is a matrix
    that is zero except on the bars that trade.

    All six size types work here, unlike the signal path — nothing has to be
    reconciled with a direction, so a target is unambiguous.

    Parameters
    ----------
    column_map
        Optional logical-column to source-column indices, as in from_signals.
        Trading parameters use the logical width; market prices retain source width.
    resume
        A portfolio to continue (``pf.update``): pass the whole history again,
        the bars it ran on and the new ones, with the same parameters, and only
        the new bars are simulated, from the portfolio's ``last_state``. The
        result is one portfolio over the whole history, equal to running it in
        one go, records in the same order. The closes of the bars it ran on
        must be the same; the starting cash is the portfolio's.
    row_offset
        With ``resume`` a ``SimState``, the low-level form a stepper uses: the
        arrays cover rows ``row_offset`` onwards of a longer history, and
        include the bars before the state's ``next_row`` that the run looks
        back at (the previous close for ``price=-inf``, ``from_ago`` bars).
        Returns a ``phasor.Segment``: the new rows' records, on the history's
        rows, and its ``last_state``. Not a portfolio: it starts mid-run.
    close
        ``(n_bars, n_cols)``, or 1-D for a single column. Sets the shape, and
        resolves the ``inf`` sentinel in ``price`` and ``val_price``.
    open, high, low
        The rest of the bar, shaped like ``close``. Only read by the
        liquidation test, which follows a path from the open to the bar's
        adverse extreme; without them it tests the close.
    size
        What to order on each bar. A number, one value per column, or a full
        matrix. Default ``inf`` — as much as the cash allows, which with the
        default price makes a buy-and-hold on the first bar.
    size_type
        ``amount``, ``value``, ``percent``, ``targetamount``, ``targetvalue``
        or ``targetpercent``. A **target** type says where the position should
        end up rather than how much to trade, which is what makes a rebalance
        one line.
    direction
        ``both`` here, rather than ``longonly`` as in
        ``from_signals``: an order's sign already says which way it goes.
    price
        ``inf`` (the default) is this bar's close, ``-inf`` the previous one.
    from_ago, bid_price, ask_price, price_area_vio_mode
        ``price`` may also name one of the bar's prices: ``"open"``,
        ``"close"``, or ``"nextopen"``/``"nextclose"``, the next bar's, which
        reads the order decided on the bar before (``from_ago=1``): a
        decision at one close executed at the following open uses nothing it
        could not have known. ``from_ago=n`` reads the order (its signals or
        size, size type and an explicit price) ``n`` bars back; a named price
        is still the executing bar's. ``from_ago`` may also be one value per
        bar (1-D) or shaped like ``close``. ``"nextvalidopen"`` and
        ``"nextvalidclose"`` skip bars without a price (``NaN`` close, or
        open): the order decided on the last bar with one executes on the
        next bar with one. ``bid_price`` and ``ask_price``, shaped
        like ``close``, are what a sell and a buy execute at where given.
        ``price_area_vio_mode`` decides what an order price outside the bar's
        low–high range does: ``"ignore"`` (the default), ``"cap"`` it to the
        range, or ``"error"``.
    force_exit
        A mask of bars that close the position at the order price in place of
        anything else on them: the end of a trading session
        (``phasor.sessions.session_end``). In ``from_signals`` a stop that
        fires inside such a bar fills first, and a resting limit order is
        cancelled.
    val_price
        What the position is marked at when a value- or percent-sized order is
        resolved. ``inf`` (the default) is the order price.
    ffill_val_price
        Carry the last usable valuation price forward.
    update_value
        Re-mark to the filled price after each fill. In shared accounts later orders in the same bar see the updated value.
    log
        As in ``from_signals``: ``True``, or a mask, to record why each order
        ended the way it did.
    cash_sharing, group_lens
        Share capital across adjacent columns. None groups all columns when
        sharing is enabled; explicit positive lengths must sum to n_cols.
        Shared init_cash is a scalar or one value per group.
    init_position, init_price
        Units held before the first bar, a number or one per column of
        ``close`` (negative: short), and the price they were opened at (NaN:
        the first close). The position is recorded as a fill on the first
        bar at that price with no fees, so it is part of the trades. A cash
        column paid for it before the run: an ``"initialposition"`` cash
        record of its cost balances the fill, the cash stays ``init_cash``,
        and the account starts worth ``init_cash`` plus the position, which
        returns count as capital brought in. A margin column holds it as an
        open trade, its margin out of free cash. No stop is armed for it.
    init_cash="auto", init_cash="autoalign"
        The cash the strategy needed: the run is made with unlimited cash,
        and each account (``"autoalign"``: every account) is then given the
        most it was ever out of pocket, net of deposits, and the run made
        again. Only for sizes that do not depend on the cash — ``amount``,
        ``value``, ``targetamount``, ``targetvalue`` — and cash accounts
        without leverage; an error if the second run fills differently.
    allow_negative_price
        Treats a price of 0 or below as a price, as a futures contract's can
        be (WTI in April 2020), rather than refusing it. Such orders must be
        sized in units (``amount``, ``targetamount``). A cash column may buy
        and sell what it holds; a margin column trades either side. Fees,
        slippage and margin are taken on the price's size, slippage always
        against the trade, and a trade's return is its profit over the size
        of its entry value. Percent stop distances, holding charges and
        liquidation assume prices above 0.
    call_seq
        "default" uses column order; "auto" stably sorts approximate order
        values, selling before buying. Neither mode allocates equal weights.
        An integer array shaped ``(n_bars, n_cols)`` gives the order itself:
        on each bar, each group's columns hold the indices within the group
        of the columns to execute first, second, and so on.
    threads
        Splits account groups, which are independent.
    cash_deposits
        Cash added (positive) or withdrawn (negative) at each bar's open,
        before orders are sized, so a target-sized order on the same bar
        invests it. A number, one value per bar (1-D), or a matrix: one column
        per account — ``(n_bars, n_cols)`` unshared, ``(n_bars, n_groups)``
        with ``cash_sharing``. A withdrawal is capped at the free cash.
        Deposits are taken out of returns: each bar's return is measured from
        the previous value plus that bar's deposit.
    cash_limit
        The most cash one buy may spend, fees included. Applied before
        ``percent`` takes its share. Sells and shorts are not affected.
    leverage, leverage_mode
        How many times its own cash a buy may be worth: 1 or more, as a
        number, one value per column or a matrix. ``lazy`` (the default)
        spends own cash first and borrows only what it cannot cover;
        ``eager`` borrows ``1 - 1/leverage`` of every buy. Fees are paid from
        own cash, and ``percent`` sizes against the levered buying power. A
        short ties up its proceeds plus ``value / leverage`` (twice its value
        at 1). A buy with no own cash left is refused as
        ``NoCashLeverage``. Under leverage ``pf.cash()`` is own cash less what
        the longs borrowed and can go negative, ``pf.value()`` stays the
        equity, and ``pf.orders()`` gains ``debt`` and ``locked_cash``.
    account_type, initial_margin, margin_basis
        ``cash`` (the default) holds what it buys; ``margin`` trades like a
        futures or perpetual contract on the account's cash. A string for
        every column or one per column, so a spot leg and a perpetual leg can
        share one account with ``cash_sharing``. Opening either side of a
        margin position posts ``initial_margin`` from free cash: a fraction of
        the notional under ``margin_basis="notional"`` (the default; ``None``
        initial margin means ``1 / leverage``), or cash per unit under
        ``"unit"``; ``margin_basis`` is a string or one per column. Every bar the
        position's profit since the last bar is settled into cash, so a loss
        shrinks what new orders may post and cash can go below zero. Closing
        realizes against the average entry and releases the margin.
        ``leverage_mode`` and ``lock_cash`` do not apply to margin columns.
        ``pf.value()`` is the equity, ``pf.free_cash()`` the equity less the
        margin posted, and ``pf.orders()`` gains ``debt`` (the cost basis),
        ``locked_cash`` (the margin) and ``margin``.
    multiplier
        What one unit of position is worth per unit of price: 50 for an
        E-mini S&P 500 future, 1 (the default) for a share. A number, or one
        value per column (per source column with ``column_map``); it cannot
        change within a column. A fill moves ``size * price * multiplier`` of
        cash, and fees, margin, dividends, holding charges and the
        liquidation test are all on that notional. Sizes, ``min_size``,
        ``size_granularity`` and a ``"unit"`` margin stay in contracts, and
        prices, stops and ``pf.orders()["price"]`` stay prices. A run with a
        multiplier other than 1 adds ``multiplier`` to ``pf.orders()``, as a
        levered run does; ``pf.trades()`` always has it, with ``pnl``
        multiplied and ``ret`` not.
    reject_prob, seed
        Probability that an order is rejected at random, for testing how a
        strategy degrades when fills fail. Each draw depends only on ``seed``,
        the bar and the column, so a run is reproducible on any thread count.
    cash_dividends
        Dividend per share, shaped like ``close`` (``column_map`` applies). Paid
        at each bar's open on the position held before that bar's orders; a
        short position pays it. Part of the return.
    cash_earnings
        Cash credited (positive) or charged (negative) at each bar's close, per
        column: interest, fees, a funding payment computed in advance. A
        number, one value per bar (1-D), or ``(n_bars, n_cols)``. Part of the
        return. A charge is capped at the free cash. ``pf.cash_records()``
        lists what was applied.
    funding_rate
        A perpetual's funding rate per bar, shaped like ``close``
        (``column_map`` applies), ``NaN`` or 0 on bars that do not settle. At
        the bar's close, after its orders, a position pays ``position * price
        * rate``: a positive rate is paid by longs and received by shorts. Put
        each rate on the bar whose close is its funding time.
    long_financing_rate, short_financing_rate
        Overnight financing per bar for each side, shaped like ``close``: a
        position pays ``abs(position) * price * rate`` at the bar's close. A
        positive rate is a cost to either side, a negative one is received.
    debt_rate
        Interest per bar on what a cash column's long borrowed under
        ``leverage``, charged at the bar's close. A number, one value per bar
        (1-D) or ``(n_bars, n_cols)``.
    borrow_rate
        The fee per bar for the shares a cash column's short borrowed, shaped
        like ``close``: ``abs(position) * price * rate`` at the bar's close.
    mark_price
        The ``price`` of the charges above, shaped like ``close``. ``None``
        uses the close.

        Every rate is per bar, for the time until the next bar;
        ``phasor.rates.per_bar`` turns an annual rate into one from the bars'
        timestamps. A charge is owed whatever the balance, so unlike a
        ``cash_earnings`` charge it is not capped: any of these runs the
        accounts as ones that may borrow, cash can go below zero, after which
        only closing is allowed, and ``pf.orders()`` gains ``debt``,
        ``locked_cash`` and ``margin``. Margin columns pay funding and
        financing but no interest or borrow fee. ``pf.cash_flows("funding")``,
        ``"financing"``, ``"interest"`` and ``"borrowfee"`` give what each
        cost per bar and column; all are part of the return.
    margin_mode, maintenance_margin, liquidation_fee
        Liquidation of margin columns. ``maintenance_margin`` (in
        ``margin_basis``: a fraction of the notional below 1, or cash per unit;
        ``None`` or 0 carries none and never liquidates) is tested on every bar
        before its orders, along a path from the bar's open to its adverse
        extreme — the low for a long, the high for a short — or to
        ``mark_price`` where one is given. An account whose equity falls
        below the maintenance margin is closed where its path first crosses
        it, or at the open when it gaps through, paying ``liquidation_fee`` of
        the notional instead of its fees. ``margin_mode`` is ``"cross"`` (the
        default: the account's equity carries every cross position and they
        close together; cash columns count as collateral and are never closed)
        or ``"isolated"`` (the position is not settled into the account's
        cash, is tested on its own margin alone and can lose no more than it
        posted), a string for every column or one per column. A loss a flat
        cross account or an isolated position cannot pay is written off, as an
        exchange's insurance fund takes it: ``pf.cash_flows("liquidation")``.
        Fills that were liquidations are flagged in ``pf.orders()`` and
        ``pf.logs()`` (``liquidation``). Without ``high`` and ``low`` the test
        uses the close. Each leg of a hedge reaches its own extreme, so a
        hedged account wants a mark price, if only the close. An isolated
        column pays its funding and financing out of its own margin.
    cash_maintenance_margin
        The maintenance margin of cash columns that borrow (margin lending:
        ``leverage`` above 1, or shorting), a fraction of the position's
        notional below 1. Below it the account's cash columns that carry one
        are liquidated with its margin columns, as a broker's margin call
        does. ``None`` (the default) keeps cash columns as collateral that is
        never closed.
    maintenance_margin_tiers
        An exchange's tier table in place of ``maintenance_margin``: a list of
        ``(notional_up_to, rate)`` pairs, increasing, the last bound ignored
        (``[(50_000, 0.004), (250_000, 0.005), (float("inf"), 0.01)]``), for
        every margin column, or one such list per column. The requirement is
        ``rate × notional − amount`` for the tier the notional falls in, the
        amount keeping it continuous at the bounds.
    liquidation_step
        The fraction of a position each breach closes, above 0 and at most 1;
        ``None`` (the default) closes it all. Below 1 the test goes on along
        the rest of the bar after each partial close, closing the same
        fraction again at each further breach.
    auto_deleverage
        A mask of bars on which the exchange auto-deleverages: a margin
        column's position is closed at ``mark_price`` (the close without one),
        with no fee, flagged as a liquidation. 1-D is one value per bar for
        every column. Which bars that hits is the caller's to say: it is the
        exchange's insurance fund, not the account, that decides.

    Returns
    -------
    Portfolio
        The run's fills, plus everything derivable from them on demand.
    """
    _resume(resume, init_cash)
    if isinstance(init_cash, str):
        return _auto_cash(from_orders, dict(locals()))
    price, price_type, from_ago = _price_option(price, from_ago, close, open, column_map)
    return _phasor.from_orders(
        _matrix(close, "close"),
        size=_param(size, "size"),
        open=None if open is None else _matrix(open, "open"),
        high=None if high is None else _matrix(high, "high"),
        low=None if low is None else _matrix(low, "low"),
        size_type=size_type,
        price=price,
        direction=direction,
        init_cash=_param(init_cash, "init_cash"),
        fees=_param(fees, "fees"),
        fixed_fees=_param(fixed_fees, "fixed_fees"),
        slippage=_param(slippage, "slippage"),
        min_size=_param(min_size, "min_size"),
        max_size=_param(max_size, "max_size"),
        size_granularity=_param(size_granularity, "size_granularity"),
        leverage=_param(leverage, "leverage"),
        leverage_mode=leverage_mode,
        account_type=_account_type(account_type),
        initial_margin=_param(initial_margin, "initial_margin"),
        margin_basis=_names(margin_basis, "margin_basis"),
        multiplier=_multiplier(multiplier, column_map),
        init_position=_multiplier(init_position, column_map, "init_position"),
        init_price=_multiplier(init_price, column_map, "init_price"),
        allow_negative_price=bool(allow_negative_price),
        price_type=price_type,
        from_ago=from_ago,
        bid_price=_market(bid_price, "bid_price"),
        ask_price=_market(ask_price, "ask_price"),
        price_area_vio_mode=price_area_vio_mode,
        force_exit=_per_bar(force_exit, close, column_map, "force_exit"),
        cash_limit=_param(cash_limit, "cash_limit"),
        reject_prob=_param(reject_prob, "reject_prob"),
        seed=seed,
        lock_cash=_flag(lock_cash, "lock_cash"),
        allow_partial=_flag(allow_partial, "allow_partial"),
        raise_reject=_flag(raise_reject, "raise_reject"),
        log=_flag(log, "log"),
        val_price=_param(val_price, "val_price"),
        cash_deposits=_param(cash_deposits, "cash_deposits"),
        cash_dividends=_market(cash_dividends, "cash_dividends"),
        cash_earnings=_param(cash_earnings, "cash_earnings"),
        funding_rate=_market(funding_rate, "funding_rate"),
        long_financing_rate=_market(long_financing_rate, "long_financing_rate"),
        short_financing_rate=_market(short_financing_rate, "short_financing_rate"),
        debt_rate=_param(debt_rate, "debt_rate"),
        borrow_rate=_market(borrow_rate, "borrow_rate"),
        mark_price=_market(mark_price, "mark_price"),
        margin_mode=_margin_mode(margin_mode),
        maintenance_margin=_param(maintenance_margin, "maintenance_margin"),
        liquidation_fee=_param(liquidation_fee, "liquidation_fee"),
        cash_maintenance_margin=_param(cash_maintenance_margin, "cash_maintenance_margin"),
        **_tiers(maintenance_margin_tiers),
        liquidation_step=_param(liquidation_step, "liquidation_step"),
        auto_deleverage=_per_bar(auto_deleverage, close, column_map, "auto_deleverage"),
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        ann_factor=ann_factor,
        risk_free=risk_free,
        trades=trades,
        incl_open=incl_open,
        threads=threads,
        cash_sharing=cash_sharing,
        group_lens=_group_lens(group_lens),
        call_seq=_call_seq(call_seq),
        column_map=_column_map(column_map),
        resume=_resume(resume, init_cash),
        row_offset=int(row_offset),
        callbacks=_callbacks,
        flexible=_flexible,
        row_wise=_row_wise,
    )


from .stepper import Stepper
from .tools import check_shapes, help
from .strategy import crossed_above, factors, strategy
