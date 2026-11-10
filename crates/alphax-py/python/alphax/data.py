"""Read-only local sources; decoding and alignment run before backtesting.

Reading is pyarrow, decoding is the caller's Python callback, and alignment is
NumPy. Install ``alphax[data]`` to read files: importing alphax and using
NumPy-only simulations does not require PyArrow.
"""
from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _alphax

InstrumentId = _alphax.InstrumentId
BarType = _alphax.BarType

_FIELDS = ("open", "high", "low", "close", "volume")
_DATA_SUFFIXES = (".csv", ".parquet", ".csv.gz", ".parquet.gz")
_GZIP = b"\x1f\x8b"
_PARQUET = b"PAR1"
_BATCH = 65536


def _arrow():
    """PyArrow, needed only when a file is read."""
    try:
        import pyarrow
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("local sources need pyarrow: pip install 'alphax-trader[data]'") from error
    return pyarrow


@dataclass(frozen=True)
class DataSchema:
    """Batch decoder contract shared by CSV, Parquet, and gzip sources.

    input/output are PyArrow schemas. Decoder receives (RecordBatch, context).
    Time columns are uint64 UTC nanoseconds. instrument is 'source', a pair of
    decoded (symbol_column, venue_column), or None for global events.
    """
    name: str
    input: object
    output: object
    decode: object
    event_time: str = "event_time"
    available_time: str = "available_time"
    instrument: object = "source"

    def __post_init__(self):
        if not callable(self.decode):
            raise TypeError("decode must be callable")
        if self.instrument not in (None, "source"):
            if not isinstance(self.instrument, tuple) or len(self.instrument) != 2 or not all(isinstance(s, str) for s in self.instrument):
                raise ValueError("instrument must be 'source', (symbol_column, venue_column), or None")


@dataclass(frozen=True)
class DecodeContext:
    """One file of one source, handed to the decoder beside every batch."""
    key: str
    path: str
    instrument: InstrumentId | None
    bar_type: BarType | None


@dataclass(frozen=True)
class BarPanel:
    time: np.ndarray
    bar_types: tuple
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @property
    def instruments(self):
        return tuple(t.instrument for t in self.bar_types)

    @property
    def shape(self):
        return self.close.shape

    @property
    def n_rows(self):
        return self.shape[0]

    @property
    def n_cols(self):
        return self.shape[1]

    def coverage(self):
        return np.isfinite(self.close).sum(axis=0) / self.n_rows if self.n_rows else np.zeros(self.n_cols)

    def drop_missing(self):
        """The panel without the times any instrument has no bar at.

        A panel of several instruments has a row for every time any of them
        traded, NaN where one did not. Dropping those rows leaves only the
        times all of them have, as ``missing_index="drop"`` does elsewhere;
        leaving them is ``"nan"``, the panel as it comes.
        """
        keep = np.isfinite(self.close).all(axis=1)
        return BarPanel(
            time=self.time[keep],
            bar_types=self.bar_types,
            open=self.open[keep],
            high=self.high[keep],
            low=self.low[keep],
            close=self.close[keep],
            volume=self.volume[keep],
        )


class Dataset:
    """Decoded Arrow batches with explicit numeric views onto a BarPanel."""
    def __init__(self, schema, keys, batches):
        self.schema = schema
        self.keys = tuple(keys)
        self.batches = tuple(batches)

    def _times(self, batch):
        return batch.column(self.schema.available_time).to_numpy(zero_copy_only=False)

    def prepare(self):
        pa = _arrow()
        bar_types = tuple(_bar_type(key) for key in self.keys)
        if self.schema.instrument is None:
            raise ValueError("bars must bind an instrument")
        for name in _FIELDS:
            if name not in self.schema.output.names or self.schema.output.field(name).type != pa.float64():
                raise ValueError(f"bar {name} must be float64")
        time = np.unique(np.concatenate([self._times(part) for _, part in self.batches])) if self.batches else np.array([], dtype=np.uint64)
        n_rows, n_cols = len(time), len(bar_types)
        fields = {name: np.full(n_rows * n_cols, np.nan) for name in _FIELDS}
        present = np.zeros(n_rows * n_cols, dtype=bool)
        for context, part in self.batches:
            column = self.keys.index(context.key)
            rows = np.searchsorted(time, self._times(part))
            if rows.size > 1:
                order = np.argsort(rows, kind="stable")
                same = np.flatnonzero(np.diff(rows[order]) == 0)
                if same.size:
                    at = order[same[0]]
                    raise ValueError(f"duplicate bar: {context.key} at {time[rows[at]]}")
            index = column * n_rows + rows
            clash = rows[present[index]]
            if clash.size:
                raise ValueError(f"duplicate bar: {context.key} at {time[clash[0]]}")
            present[index] = True
            for name, values in fields.items():
                values[index] = part.column(name).to_numpy(zero_copy_only=False)
        return BarPanel(
            time=time,
            bar_types=bar_types,
            **{name: _fortran(values, n_rows, n_cols) for name, values in fields.items()},
        )

    def _row_groups(self, part):
        """The distinct instruments of one batch, and each row's index into them."""
        context, batch = part
        binding = self.schema.instrument
        if binding is None:
            return (None,), None
        if binding == "source":
            return (context.instrument,), None
        rows = batch.num_rows
        if not rows:
            return (), np.zeros(0, dtype=np.int64)
        pairs = np.array(list(zip(batch.column(binding[0]).to_pylist(), batch.column(binding[1]).to_pylist())))
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        return tuple(InstrumentId(str(symbol), str(venue)) for symbol, venue in unique), inverse

    def _events(self, field):
        """(times, values) per instrument, each ordered by time, ties included."""
        pa = _arrow()
        if field not in self.schema.output.names:
            raise ValueError(f"{field} is not in the output schema")
        dtype = self.schema.output.field(field).type
        if not (pa.types.is_floating(dtype) or pa.types.is_integer(dtype)):
            raise ValueError(f"{field} must be numeric to align")
        grouped = {}
        for part in self.batches:
            context, batch = part
            times = self._times(batch)
            values = _floats(batch.column(field))
            instruments, inverse = self._row_groups(part)
            if inverse is None:
                grouped.setdefault(instruments[0], []).append((times, values))
                continue
            for at, instrument in enumerate(instruments):
                mask = inverse == at
                grouped.setdefault(instrument, []).append((times[mask], values[mask]))
        events = {}
        for instrument, parts in grouped.items():
            times = np.concatenate([times for times, _ in parts])
            values = np.concatenate([values for _, values in parts])
            order = np.argsort(times, kind="stable")
            events[instrument] = (times[order], values[order])
        return events

    def _align(self, time, instruments, field, window=None, agg="mean", empty=np.nan):
        """Latest available value per time, or an aggregate over (t-window, t]."""
        target = np.asarray(time)
        if target.ndim != 1 or (target.size > 1 and np.any(np.diff(target) <= 0)):
            raise ValueError("target time must be strictly increasing")
        if window is not None and window <= 0:
            raise ValueError("window must be positive")
        if agg not in ("mean", "sum", "count"):
            raise ValueError("agg must be mean, sum, or count")
        events = self._events(field)
        packed = np.full(target.size * len(instruments), np.nan)
        for column, instrument in enumerate(instruments):
            series = events.get(None if self.schema.instrument is None else instrument)
            if series is None:
                continue
            times, values = series
            if not times.size:
                continue
            right = np.searchsorted(times, target, side="right")
            if window is None:
                has = right > 0
                packed[column * target.size : (column + 1) * target.size] = np.where(
                    has, values[np.where(has, right - 1, 0)], np.nan
                )
                continue
            left = np.zeros_like(right)
            bounded = target >= window
            left[bounded] = np.searchsorted(times, target[bounded] - window, side="right")
            finite = np.isfinite(values)
            sums = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
            counts = np.concatenate(([0], np.cumsum(finite)))
            if agg == "count":
                result = (right - left).astype(float)
            else:
                seen = counts[right] - counts[left]
                total = sums[right] - sums[left]
                if agg == "mean":
                    total = np.divide(total, seen, out=np.zeros_like(total), where=seen > 0)
                result = np.where(seen > 0, total, empty)
            packed[column * target.size : (column + 1) * target.size] = result
        return packed.reshape((target.size, len(instruments)), order="F")

    def asof(self, panel, *, field):
        """Latest available value; missing history remains NaN. Ties use input order."""
        return self._align(panel.time, list(panel.instruments), field)

    def window(self, panel, *, field, window, agg="mean", empty=np.nan):
        """Aggregate (t-window, t] by availability. Count counts all event rows.

        Sum/mean ignore null and nonfinite values; empty supplies their result
        when there are no finite observations. Global events broadcast.
        """
        if isinstance(window, str):
            match = re.fullmatch(r"([1-9][0-9]*)(ns|us|ms|s|m|h|d|w)", window)
            if not match:
                raise ValueError("window must be positive nanoseconds or a duration such as '30m'")
            units = dict(ns=1, us=1000, ms=1000000, s=10**9, m=60*10**9, h=3600*10**9, d=86400*10**9, w=604800*10**9)
            window = int(match[1]) * units[match[2]]
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window <= 0:
            raise ValueError("window must be positive nanoseconds")
        return self._align(panel.time, list(panel.instruments), field, int(window), agg, empty)


def _fortran(values, n_rows, n_cols):
    return values.reshape((n_rows, n_cols), order="F")


def _floats(column):
    """A float64 view of an Arrow column, with nulls and nonfinite values as NaN."""
    pa = _arrow()
    if column.type != pa.float64():
        column = column.cast(pa.float64())
    return column.to_numpy(zero_copy_only=False)


def _project(batch, schema):
    columns = []
    for field in schema:
        if field.name not in batch.schema.names:
            raise ValueError(f"missing field {field.name}")
        column = batch.column(field.name)
        if column.type != field.type or (not field.nullable and column.null_count):
            raise ValueError(f"field {field.name} does not match declared type/nullability")
        columns.append(column)
    return _arrow().RecordBatch.from_arrays(columns, schema=schema)


def _data_file(name):
    """Anything a scan reads: plain or gzipped CSV/Parquet, never hidden."""
    lowered = name.lower()
    return not name.startswith(".") and lowered.endswith(_DATA_SUFFIXES)


def _scan(directory):
    paths = sorted((path for path in directory.iterdir() if path.is_file() and _data_file(path.name)), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"no CSV or Parquet files in {directory}")
    return paths


def _component(value):
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("keys and instruments must be single path components")


def _resolve(path, keys):
    """The files each key reads, in the order they are read.

    A path is one file, a directory holding one key's files, or a directory
    whose children are named after the keys.
    """
    if not keys:
        raise ValueError("a source needs at least one key")
    if path.is_file():
        if len(keys) != 1:
            raise ValueError(f"{path} is a single file, so it holds one key, not {len(keys)}")
        _component(keys[0])
        return [(keys[0], [path])]
    if not path.is_dir():
        raise ValueError(f"{path} is neither a file nor a directory")
    items = []
    for key in keys:
        _component(key)
        joined = path / key
        if joined.is_file():
            paths = [joined]
        elif joined.is_dir():
            paths = _scan(joined)
        elif len(keys) == 1:
            paths = _scan(path)
        else:
            raise ValueError(f"no files for {key} in {path}; pass the directory holding them")
        items.append((key, paths))
    return items


def _format(path):
    """Whether a file is gzipped, and whether it holds Parquet, by content."""
    with path.open("rb") as handle:
        gzipped = handle.read(2) == _GZIP
    try:
        opener = gzip.open if gzipped else Path.open
        with opener(path, "rb") as handle:
            return gzipped, handle.read(4) == _PARQUET
    except (OSError, EOFError) as error:
        raise ValueError(f"not a readable gzip or Parquet file: {error}") from error


def _read(path, schema):
    """Batches of the declared input columns; content, not the name, decides."""
    pa = _arrow()
    gzipped, parquet = _format(path)
    names = schema.names
    try:
        if parquet:
            if gzipped:
                raise ValueError("gzipped Parquet is not supported: store Parquet uncompressed, it compresses internally")
            import pyarrow.parquet as parquet_reader
            yield from parquet_reader.ParquetFile(path).iter_batches(batch_size=_BATCH, columns=names)
            return
        import pyarrow.csv as csv_reader
        options = csv_reader.ConvertOptions(column_types=schema, include_columns=names)
        stream = gzip.open(path, "rb") if gzipped else str(path)
        yield from csv_reader.open_csv(stream, convert_options=options)
    except (pa.ArrowException, OSError, EOFError) as error:
        # A missing column, a truncated stream or a bad header is data, not Arrow.
        raise ValueError(str(error)) from error


def _bar_type(key):
    try:
        return BarType.parse(key)
    except ValueError as error:
        raise ValueError(f"{key} is not a BarType: {error}") from error


def _parse_instrument(key):
    try:
        return InstrumentId.parse(key)
    except ValueError:
        return None


def _key_instrument(key):
    """The instrument a key names: its BarType, or a bare instrument identity."""
    try:
        bar_type = BarType.parse(key)
    except ValueError:
        return None, _parse_instrument(key)
    return bar_type, bar_type.instrument


def _pairs(batch, symbol, venue):
    """One InstrumentId per row of a batch that carries its own symbols."""
    for name in (symbol, venue):
        if batch.column(name).null_count:
            raise ValueError(f"{name} cannot contain nulls")
    return [InstrumentId(s, v) for s, v in zip(batch.column(symbol).to_pylist(), batch.column(venue).to_pylist())]


def _window(batch, name, start, end):
    """The rows whose availability falls in [start, end)."""
    time = batch.column(name).to_numpy(zero_copy_only=False)
    keep = np.ones(len(time), dtype=bool)
    if start is not None:
        keep &= time >= start
    if end is not None:
        keep &= time < end
    return batch if keep.all() else batch.take(np.flatnonzero(keep))


def _load(schema, items, start, end):
    """Read every file, decode each batch, and keep what falls in [start, end)."""
    if not items or (start is not None and end is not None and start >= end):
        raise ValueError("empty source or invalid [start, end) interval")
    batches = []
    seen = set()
    for key, paths in items:
        if not paths or key in seen:
            raise ValueError("source keys must be unique and have files")
        seen.add(key)
        bar_type, instrument = _key_instrument(key)
        if schema.instrument == "source" and instrument is None:
            raise ValueError(f"invalid instrument or BarType key: {key}")
        for path in paths:
            context = DecodeContext(key=key, path=str(path), instrument=instrument, bar_type=bar_type)
            try:
                for raw in _read(path, schema.input):
                    batch = _project(schema.decode(_project(raw, schema.input), context), schema.output)
                    for name in (schema.event_time, schema.available_time):
                        if batch.column(name).null_count:
                            raise ValueError(f"{name} cannot contain nulls")
                    if isinstance(schema.instrument, tuple):
                        pairs = _pairs(batch, schema.instrument[0], schema.instrument[1])
                        if bar_type is not None and any(one != instrument for one in pairs):
                            raise ValueError("bar row instrument differs from source BarType")
                    batches.append((context, _window(batch, schema.available_time, start, end)))
            except (ValueError, OSError) as error:
                raise ValueError(f"{path} [{key}]: {error}") from error
    return batches


def _paths(value):
    """One path or an ordered list of them."""
    if isinstance(value, (str, Path)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    raise TypeError("a source takes a path or a sequence of paths")


def _inferred(path):
    """The key a path names, for callers that do not pass `instruments`.

    Only the last path component is read, and only as an instrument: a file
    name never carries data. A directory of directories has no single
    instrument, so it needs `instruments` explicitly.
    """
    name = Path(path).expanduser().name
    if not name:
        raise ValueError("pass instruments: this path names none")
    if _parse_instrument(name) is None and not _is_bar_type(name):
        raise ValueError(f"pass instruments: {name!r} is neither a BarType nor an InstrumentId")
    return [name]


def _is_bar_type(value):
    try:
        BarType.parse(value)
    except ValueError:
        return False
    return True


def _nanoseconds(value, *, name):
    """A bound of the read interval, as uint64 UTC nanoseconds or None."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be nanoseconds or an ISO-8601 timestamp")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(np.datetime64(value, "ns").astype(np.int64))
        except ValueError as error:
            raise ValueError(f"{name} must be nanoseconds or an ISO-8601 timestamp: {value!r}") from error
    raise TypeError(f"{name} must be nanoseconds or an ISO-8601 timestamp")


class Source:
    """Files and a decoder, resolved from one path; loading is explicit and writes nothing.

    ``path`` is a file, a directory holding one key's files, or a directory
    whose children are named after the keys. A mapping names the keys itself
    instead, with a path or an ordered list of paths per key. Only a file's
    content decides whether it is CSV or Parquet, and gzip is transparent.
    """
    def __init__(self, path, schema, instruments=None):
        self.schema = schema
        if isinstance(path, Mapping):
            if instruments is not None:
                raise TypeError("a mapping already names its keys; drop instruments")
            items = [(str(key), [Path(one).expanduser() for one in _paths(value)]) for key, value in path.items()]
        else:
            keys = _inferred(path) if instruments is None else (
                [instruments] if isinstance(instruments, (str, Path)) else [str(key) for key in instruments]
            )
            items = _resolve(Path(path).expanduser(), [str(key) for key in keys])
        self._sources = tuple((str(key), tuple(paths)) for key, paths in items)

    @property
    def items(self):
        """The (key, paths) pairs this source resolved, with paths as strings."""
        return tuple((key, tuple(str(path) for path in paths)) for key, paths in self._sources)

    def load(self, *, start=None, end=None):
        """Read [start,end) in decoded availability nanoseconds, or ISO-8601.

        Load auxiliary history before the target panel when using asof/windows;
        an explicit start discards earlier observations.
        """
        keys = [key for key, _ in self._sources]
        return Dataset(self.schema, keys, _load(
            self.schema,
            self._sources,
            _nanoseconds(start, name="start"),
            _nanoseconds(end, name="end"),
        ))

    def prepare(self, *, start=None, end=None):
        return self.load(start=start, end=end).prepare()


def load(path, schema, instruments=None, *, start=None, end=None):
    """Read one path into a `Dataset`; `prepare()` then gives the bar panel.

    `path` is a file, a directory holding one key's files, or a directory whose
    children are named after the keys. `instruments` is one key or a list of
    them, in panel-column order; it is inferred from the directory name when it
    is a single key's directory. `start` and `end` bound the decoded
    availability column half open, as uint64 UTC nanoseconds or ISO-8601.
    """
    return Source(path, schema, instruments).load(
        start=_nanoseconds(start, name="start"), end=_nanoseconds(end, name="end")
    )
