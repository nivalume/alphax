"""Qlib's data processors over factor panels.

The processors of ``qlib.data.dataset.processor`` with Qlib's names,
arguments and arithmetic, over what :meth:`alphax.FactorSet.evaluate`
returns: a mapping of names to ``(rows, symbols)`` matrices, one row per bar
time. Labels sit in the same mapping (``alphax.qlib_label()`` names its
``LABEL0``).

``fields_group`` picks the fields a processor touches: ``None`` is every
field, ``"label"`` those whose name contains ``LABEL`` (Qlib's label names),
``"feature"`` the rest, or a list of names.

Two kinds:

- **Stateless**, per row or per cell: :class:`ProcessInf`, :class:`Fillna`,
  :class:`CSZScoreNorm`, :class:`CSRankNorm`, :class:`CSZFillna`,
  :class:`TanhProcess`, :class:`DropnaProcessor`, :class:`DropnaLabel`,
  :class:`DropCol`, :class:`FilterCol`.
- **Fitted** on a time range: :class:`ZScoreNorm`, :class:`RobustZScoreNorm`,
  :class:`MinMaxNorm` learn each field's statistics over the rows from
  ``fit_start_time`` to ``fit_end_time`` (inclusive; a date alone covers its
  whole day) and apply them to every row. **The range must end before any
  data you evaluate on**, or the statistics carry the test period into
  training. ``fit`` needs ``index``, the time of each row.

Qlib drops rows (``DropnaLabel``, ``DropnaProcessor``); a panel keeps them, so
a dropped row is ``NaN`` in every field instead. Every statistic here skips
``NaN``, so what follows sees exactly what it would see in Qlib; to train,
keep the cells whose label is not ``NaN``.

:func:`process` runs a list as Qlib's ``DataHandlerLP`` does: each processor
fitted on the output of the one before, then applied::

    infer = [ProcessInf(), ZScoreNorm("2015-01-01", "2019-12-31"), Fillna()]
    data = process({**features, **labels}, infer, index=dates)
    learn = process(data, [DropnaLabel(), CSZScoreNorm(fields_group="label")])
    later = process(new_features, infer, index=new_dates, fit=False)  # same statistics

Nothing changes its input: a call returns a new mapping, sharing the arrays
of the fields it leaves alone.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "Processor",
    "process",
    "ProcessInf",
    "Fillna",
    "ZScoreNorm",
    "RobustZScoreNorm",
    "MinMaxNorm",
    "CSZScoreNorm",
    "CSRankNorm",
    "CSZFillna",
    "TanhProcess",
    "DropnaProcessor",
    "DropnaLabel",
    "DropCol",
    "FilterCol",
]

# Qlib's EPS, added to RobustZScoreNorm's spread.
EPS = 1e-12
_NANOS_PER_DAY = 86_400_000_000_000

Panel = dict[str, np.ndarray]


@contextmanager
def _quiet() -> Iterator[None]:
    """NumPy's warnings for an all-NaN slice or a division by zero, which the
    results already say with NaN and infinity."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _fields(data: Mapping[str, np.ndarray], group: Any) -> list[str]:
    if group is None:
        return list(data)
    if isinstance(group, str):
        if group == "label":
            return [name for name in data if "LABEL" in name]
        if group == "feature":
            return [name for name in data if "LABEL" not in name]
        raise ValueError(f"fields_group must be None, 'feature', 'label' or a list of names; got {group!r}")
    names = [str(name) for name in group]
    missing = [name for name in names if name not in data]
    if missing:
        raise KeyError(f"fields {missing} are not in the data")
    return names


def _panel(data: Mapping[str, Any]) -> Panel:
    if not isinstance(data, Mapping):
        raise TypeError("processor data must be a mapping of names to (rows, symbols) arrays")
    out = {}
    shape = None
    for name, values in data.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"field {name!r} has shape {array.shape}, expected {shape}")
        out[name] = array
    return out


def _instant(value: Any, end: bool) -> int:
    """A fit bound as wall-clock nanoseconds. A date alone (``"2020-02-05"``)
    covers its whole day, as pandas' partial-string slicing does."""
    if getattr(value, "tzinfo", None) is not None:
        value = value.replace(tzinfo=None)
    nanos = int(np.datetime64(value, "ns").astype(np.int64))
    if end and isinstance(value, str) and len(value.strip()) == 10:
        nanos += _NANOS_PER_DAY - 1
    return nanos


def _fit_rows(index: Any, start: Any, end: Any, n_rows: int) -> np.ndarray:
    if index is None:
        raise ValueError("fitting on a time range needs index=, the time of each row")
    from . import _timestamps

    times = _timestamps(index)
    if times.shape[0] != n_rows:
        raise ValueError(f"index has {times.shape[0]} rows, the data {n_rows}")
    rows = np.ones(n_rows, dtype=bool)
    if start is not None:
        rows &= times >= _instant(start, end=False)
    if end is not None:
        rows &= times <= _instant(end, end=True)
    return rows


def _cross(template: str, data: Panel, names: list[str]) -> Panel:
    """``template`` (with ``{}`` for the field) per field, through the
    engine's cross-sectional operators, whose ranks and moments are pandas'."""
    if not names:
        return {}
    from . import FactorSet

    alias = {name: f"f{k}" for k, name in enumerate(names)}
    factors = FactorSet.compile({alias[name]: template.format(f"${alias[name]}") for name in names})
    values = factors.evaluate({alias[name]: data[name] for name in names})
    return {name: values[alias[name]] for name in names}


class Processor:
    """A step of Qlib's data pipeline. ``fit`` learns parameters (a no-op for
    the stateless ones); calling the processor applies them."""

    def fit(self, data: Mapping[str, Any], index: Any = None) -> Processor:
        return self

    def __call__(self, data: Mapping[str, Any]) -> Panel:
        raise NotImplementedError

    def is_for_infer(self) -> bool:
        """Whether it may run at inference time; a processor that drops rows by
        the label may not."""
        return True

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


def process(
    data: Mapping[str, Any],
    processors: Sequence[Processor],
    index: Any = None,
    fit: bool = True,
) -> Panel:
    """Runs ``processors`` in order over ``data``, as Qlib's ``DataHandlerLP``:
    with ``fit``, each is fitted on the previous one's output first. Without
    it, already fitted processors apply what they learned."""
    out = _panel(data)
    for processor in processors:
        if fit:
            processor.fit(out, index)
        out = processor(out)
    return out


class ProcessInf(Processor):
    """Each infinity replaced by the mean of its row's finite values in the
    same field (Qlib's per-datetime replacement); ``NaN`` when there are none."""

    def __call__(self, data):
        out = _panel(data)
        with _quiet():
            for name, x in out.items():
                infinite = np.isinf(x)
                if infinite.any():
                    mean = np.nanmean(np.where(infinite, np.nan, x), axis=1, keepdims=True)
                    out[name] = np.where(infinite, mean, x)
        return out


class Fillna(Processor):
    """``NaN`` replaced by ``fill_value``."""

    def __init__(self, fields_group: Any = None, fill_value: float = 0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, data):
        out = _panel(data)
        for name in _fields(out, self.fields_group):
            out[name] = np.where(np.isnan(out[name]), self.fill_value, out[name])
        return out


class _Fitted(Processor):
    """Per-field statistics over the fit range, applied to every row."""

    def __init__(self, fit_start_time: Any, fit_end_time: Any, fields_group: Any = None):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self._stats: dict[str, tuple[float, float]] | None = None

    def _statistics(self, block: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError

    def fit(self, data, index=None):
        panel = _panel(data)
        n_rows = next(iter(panel.values())).shape[0] if panel else 0
        rows = _fit_rows(index, self.fit_start_time, self.fit_end_time, n_rows)
        with _quiet():
            self._stats = {
                name: self._statistics(panel[name][rows]) for name in _fields(panel, self.fields_group)
            }
        return self

    def _apply(self, x: np.ndarray, centre: float, scale: float) -> np.ndarray:
        return (x - centre) / scale

    def __call__(self, data):
        if self._stats is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before it is applied")
        out = _panel(data)
        with _quiet():
            for name, (centre, scale) in self._stats.items():
                out[name] = self._apply(out[name], centre, scale)
        return out


class ZScoreNorm(_Fitted):
    """``(x - mean) / std`` with the fit range's mean and population standard
    deviation; a field constant over the range is left alone."""

    def _statistics(self, block):
        mean, std = np.nanmean(block), np.nanstd(block)
        return (0.0, 1.0) if std == 0 else (mean, std)


class RobustZScoreNorm(_Fitted):
    """``(x - median) / (1.4826 · (MAD + EPS))`` with the fit range's median
    and median absolute deviation, clipped to ``[-3, 3]`` unless
    ``clip_outlier`` is false."""

    def __init__(self, fit_start_time: Any, fit_end_time: Any, fields_group: Any = None, clip_outlier: bool = True):
        super().__init__(fit_start_time, fit_end_time, fields_group)
        self.clip_outlier = clip_outlier

    def _statistics(self, block):
        median = np.nanmedian(block)
        return median, (np.nanmedian(np.abs(block - median)) + EPS) * 1.4826

    def _apply(self, x, centre, scale):
        x = (x - centre) / scale
        return np.clip(x, -3, 3) if self.clip_outlier else x


class MinMaxNorm(_Fitted):
    """``(x - min) / (max - min)`` with the fit range's extremes; a field
    constant over the range is left alone."""

    def _statistics(self, block):
        low, high = np.nanmin(block), np.nanmax(block)
        return (0.0, 1.0) if low == high else (low, high - low)


class CSZScoreNorm(Processor):
    """Each row standardised across symbols: ``(x - mean) / std`` (sample
    standard deviation), or with ``method="robust"`` ``x - median`` over
    ``1.4826 · MAD``, clipped to ``[-3, 3]``."""

    def __init__(self, fields_group: Any = None, method: str = "zscore"):
        if method not in ("zscore", "robust"):
            raise ValueError("method must be 'zscore' or 'robust'")
        self.fields_group = fields_group
        self.method = method

    def __call__(self, data):
        out = _panel(data)
        names = _fields(out, self.fields_group)
        if self.method == "zscore":
            out.update(_cross("CSZScore({})", out, names))
            return out
        with _quiet():
            for name in names:
                x = out[name] - np.nanmedian(out[name], axis=1, keepdims=True)
                mad = np.nanmedian(np.abs(x), axis=1, keepdims=True)
                out[name] = np.clip(x / mad / 1.4826, -3, 3)
        return out


class CSRankNorm(Processor):
    """Each row's percentile rank across symbols (ties averaged), centred and
    scaled towards unit spread: ``(rank - 0.5) · 3.46``."""

    def __init__(self, fields_group: Any = None):
        self.fields_group = fields_group

    def __call__(self, data):
        out = _panel(data)
        out.update(_cross("(CSRank({}) - 0.5) * 3.46", out, _fields(out, self.fields_group)))
        return out


class CSZFillna(Processor):
    """``NaN`` replaced by its row's mean across symbols."""

    def __init__(self, fields_group: Any = None):
        self.fields_group = fields_group

    def __call__(self, data):
        out = _panel(data)
        with _quiet():
            for name in _fields(out, self.fields_group):
                mean = np.nanmean(out[name], axis=1, keepdims=True)
                out[name] = np.where(np.isnan(out[name]), mean, out[name])
        return out


class TanhProcess(Processor):
    """``tanh(x - 1)`` on every field that is not a label, exactly as Qlib's
    (which subtracts the 1)."""

    def __call__(self, data):
        out = _panel(data)
        for name in _fields(out, "feature"):
            out[name] = np.tanh(out[name] - 1)
        return out


class DropnaProcessor(Processor):
    """Drops the rows where a field of ``fields_group`` is ``NaN``: every
    field becomes ``NaN`` there."""

    def __init__(self, fields_group: Any = None):
        self.fields_group = fields_group

    def __call__(self, data):
        out = _panel(data)
        names = _fields(out, self.fields_group)
        if not names:
            return out
        dropped = np.logical_or.reduce([np.isnan(out[name]) for name in names])
        return {name: np.where(dropped, np.nan, x) for name, x in out.items()}


class DropnaLabel(DropnaProcessor):
    """Drops the rows whose label is ``NaN``. Training only."""

    def __init__(self, fields_group: Any = "label"):
        super().__init__(fields_group)

    def is_for_infer(self) -> bool:
        return False


class DropCol(Processor):
    """Removes the fields named in ``col_list``."""

    def __init__(self, col_list: Sequence[str] = ()):
        self.col_list = list(col_list)

    def __call__(self, data):
        return {name: x for name, x in _panel(data).items() if name not in self.col_list}


class FilterCol(Processor):
    """Keeps only the fields of ``fields_group`` named in ``col_list``, and
    every field outside the group."""

    def __init__(self, fields_group: Any = "feature", col_list: Sequence[str] = ()):
        self.fields_group = fields_group
        self.col_list = list(col_list)

    def __call__(self, data):
        out = _panel(data)
        group = set(_fields(out, self.fields_group))
        return {name: x for name, x in out.items() if name not in group or name in self.col_list}
