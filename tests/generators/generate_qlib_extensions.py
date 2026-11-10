"""Generate the Qlib oracles for alphax-expr's Qlib extensions.

Writes four fixtures:

* ``qlib_alpha158_configs.tsv``: the names and expression strings
  ``Alpha158DL.get_feature_config(config)`` produces for several configs.
* ``qlib_highfreq.tsv``: Qlib's high-frequency operators
  (``qlib.contrib.ops.high_freq``) and the cross-instrument operators
  ``ChangeInstrument`` and ``Mask`` over a 30-minute panel.
* ``qlib_semantics.tsv``: the corners of Qlib's operator semantics that
  depend on Python: ``Ref(x, 0)`` / ``Delta(x, 0)`` (the first row), windows
  strictly between 0 and 1 (an exponential mean), arithmetic in literal
  arguments, ``True`` / ``False``, and Unicode field names.
* ``qlib_pit_records.tsv`` and ``qlib_pit.tsv``: point-in-time financial
  records and ``P`` / ``PRef`` evaluated through Qlib's own
  ``LocalPITProvider`` reading real PIT binary files.

As in the Alpha158 generator, only Qlib's leaves are replaced: the bar
feature provider and the high-frequency calendar. The parser, the expression
tree, the cache, the operators and the PIT provider are Qlib's. The PIT files
are written in the layout of Qlib's ``scripts/dump_pit.py`` into a temporary
provider directory, never into the repository.

    conda run -n rdagent4qlib python tests/generators/generate_qlib_extensions.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.config import C
from qlib.contrib.data.loader import Alpha158DL
from qlib.contrib.ops import high_freq
from qlib.contrib.ops.high_freq import BFillNan, Cut, Date, DayCumsum, DayLast, FFillNan, IsInf, IsNull, Select
from qlib.data.base import Feature
from qlib.data.cache import H
from qlib.data.data import LocalExpressionProvider
from qlib.utils import get_period_offset

FIXTURES = Path(__file__).parents[1] / "fixtures"
EPOCH = dt.date(1970, 1, 1)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, ".17g")


def versions(handle) -> None:
    handle.write(f"# qlib={qlib.__version__}\n")
    handle.write(f"# numpy={np.__version__} pandas={pd.__version__}\n")


# --------------------------------------------------------------------------
# Alpha158 configs
# --------------------------------------------------------------------------

CONFIGS: list[tuple[str, dict | None]] = [
    ("default", None),
    (
        "full",
        {
            "kbar": {},
            "price": {"windows": [0, 1, 2, 3, 4], "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]},
            "volume": {"windows": [0, 1, 2, 3, 4]},
            "rolling": {},
        },
    ),
    ("implicit", {"price": {}, "volume": {}}),
    (
        "subset",
        {"rolling": {"windows": [3, 7], "include": ["ROC", "MA", "LOW", "CORD", "VSUMD"], "exclude": ["MA"]}},
    ),
    ("exclude", {"kbar": {}, "rolling": {"exclude": ["RANK", "CORR", "CORD"]}}),
    ("price_only", {"price": {"windows": [2, 0], "feature": ["close", "Vwap"]}}),
]


def write_alpha158_configs() -> None:
    out = FIXTURES / "qlib_alpha158_configs.tsv"
    with out.open("w", newline="") as handle:
        versions(handle)
        for name, config in CONFIGS:
            text = "qlib default" if config is None else json.dumps(config, sort_keys=True)
            handle.write(f"# config {name} = {text}\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["config", "position", "name", "expression"])
        total = 0
        for name, config in CONFIGS:
            if config is None:
                fields, names = Alpha158DL.get_feature_config()
            else:
                fields, names = Alpha158DL.get_feature_config(config)
            assert len(fields) == len(names)
            for position, (factor, expression) in enumerate(zip(names, fields, strict=True)):
                assert "\t" not in expression
                writer.writerow([name, position, factor, expression])
            total += len(names)
    print(f"wrote {out} ({len(CONFIGS)} configs, {total} factors)")


# --------------------------------------------------------------------------
# High-frequency and cross-instrument operators
# --------------------------------------------------------------------------

HF_SYMBOLS = ("AAA", "BBB")
HF_TIMES = ("09:30", "10:00", "10:30", "11:00", "13:00", "13:30", "14:00", "14:30")
HF_DAYS = pd.bdate_range("2020-01-02", periods=10)
HF_ROWS = len(HF_DAYS) * len(HF_TIMES)
HF_INPUTS = ("close", "volume", "paused_num", "x")
HF_EXPRESSIONS = (
    "DayCumsum($volume, '9:30', '14:59', 30)",
    "DayCumsum($volume, '10:15', '13:44', 30)",
    "DayCumsum($volume, '10:00', '11:00', 30)",
    "DayCumsum($x, '9:30', '14:59', 30)",
    "DayLast($close)",
    "DayLast(Ref($close, 9))",
    "$close/DayLast(Ref($close, 9))",
    "FFillNan($close)",
    "BFillNan($close)",
    "IsNull($close)",
    "IsInf($x)",
    "If(IsInf($x), 0, $x)",
    "If(IsNull($close), 0, $close)",
    "Select(Gt($paused_num, 1.001), $close)",
    "Cut($close, 16, None)",
    "Cut($close, 3, -2)",
    "Cut(FFillNan($close)/DayLast(Ref(FFillNan($close), 16)), 16, None)",
    "Ref(DayLast(Mean($volume, 40)), 8)",
    "Date($close)",
    "ChangeInstrument('BBB', $close)",
    "$close/ChangeInstrument('BBB', $close)",
    "Mask($close, 'BBB')",
    "ChangeInstrument('BBB', DayLast($close))",
    "Corr($close, ChangeInstrument('BBB', $close), 10)",
)

# Awkward rows, by symbol. Rows are day * 8 + bar.
HF_CLOSE_NAN = {
    # first bar of day 2, a mid-day bar of day 3, the last bar of day 4, the
    # very last row (nothing to back-fill from)
    "AAA": [16, 27, 39, 79],
    # the first row (nothing to forward-fill from), first bar of day 2, the
    # last bar of day 3, and all of day 6
    "BBB": [0, 16, 31, *range(48, 56)],
}
HF_VOLUME_NAN = {"AAA": [21], "BBB": [50]}
HF_PAUSED = {"AAA": {5: 1.0, 33: 1.001, 60: math.nan}, "BBB": {12: 1.0, 70: 1.0005}}
HF_X = {
    "AAA": {3: math.inf, 10: -math.inf, 11: math.nan, 44: math.inf},
    # -inf then +inf on the same day: the day's running sum becomes NaN
    "BBB": {2: -math.inf, 5: math.inf, 30: math.inf, 31: math.nan},
}


def hf_panel() -> tuple[list[dt.datetime], dict[str, dict[str, np.ndarray]]]:
    stamps = [
        dt.datetime.combine(day.date(), dt.time.fromisoformat(time)) for day in HF_DAYS for time in HF_TIMES
    ]
    data: dict[str, dict[str, np.ndarray]] = {}
    for s, symbol in enumerate(HF_SYMBOLS):
        row = np.arange(HF_ROWS, dtype=np.float64)
        close = 10.0 + s * 5.0 + 0.05 * row + 0.3 * np.sin(0.7 * row + s)
        volume = 1_000.0 + 37.0 * ((row * 7 + s * 3) % 11) + 5.0 * row
        paused = np.full(HF_ROWS, 2.0)
        x = np.round(np.sin(0.9 * row + 2 * s) * 3.0, 6)
        close[HF_CLOSE_NAN[symbol]] = np.nan
        volume[HF_VOLUME_NAN[symbol]] = np.nan
        for at, value in HF_PAUSED[symbol].items():
            paused[at] = value
        for at, value in HF_X[symbol].items():
            x[at] = value
        data[symbol] = {"close": close, "volume": volume, "paused_num": paused, "x": x}
    return stamps, data


def as_float(values: pd.Series) -> np.ndarray:
    """Qlib output on the full row range: booleans as 1/0, dates as days
    since 1970-01-01, rows Select/Cut dropped as NaN."""
    values = values.reindex(pd.RangeIndex(0, HF_ROWS))
    if any(isinstance(v, dt.date) for v in values):
        return np.array([(v - EPOCH).days if isinstance(v, dt.date) else math.nan for v in values])
    return values.to_numpy(dtype=np.float64)


def write_highfreq() -> None:
    stamps, data = hf_panel()
    dates = np.array([stamp.date() for stamp in stamps])

    def load_feature(self: Feature, instrument: str, start: int, end: int, *_args) -> pd.Series:
        return pd.Series(data[instrument][self._name][start : end + 1], index=pd.RangeIndex(start, end + 1))

    Feature._load_internal = load_feature
    high_freq.get_calendar_day = lambda freq="1min", future=False: dates

    provider = LocalExpressionProvider(time2idx=False)
    output: dict[str, dict[str, np.ndarray]] = {symbol: {} for symbol in HF_SYMBOLS}
    for expression in HF_EXPRESSIONS:
        # DayCumsum and Cut print as `DayCumsum($volume)` / `Cut($close)`,
        # without their time or cut arguments, and that string is Qlib's
        # memory-cache key: a second DayCumsum over the same feature would
        # get the first one's cached series. Each expression starts cold.
        H["f"].clear()
        for symbol in HF_SYMBOLS:
            values = provider.get_expression_instance(expression).load(symbol, 0, HF_ROWS - 1, "1min")
            output[symbol][expression] = as_float(values)

    out = FIXTURES / "qlib_highfreq.tsv"
    with out.open("w", newline="") as handle:
        versions(handle)
        handle.write(
            f"# layout=symbol-major rows={HF_ROWS} symbols={','.join(HF_SYMBOLS)} "
            f"bars/day={len(HF_TIMES)} granularity=30min\n"
        )
        handle.write("# booleans are 1/0; Date() is days since 1970-01-01; rows Select/Cut drop are nan\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["symbol", "row", "datetime", *HF_INPUTS, *HF_EXPRESSIONS])
        for symbol in HF_SYMBOLS:
            for row, stamp in enumerate(stamps):
                writer.writerow(
                    [
                        symbol,
                        row,
                        stamp.isoformat(),
                        *(format_float(float(data[symbol][name][row])) for name in HF_INPUTS),
                        *(format_float(float(output[symbol][e][row])) for e in HF_EXPRESSIONS),
                    ]
                )
    print(f"wrote {out} ({len(HF_SYMBOLS) * HF_ROWS} rows, {len(HF_EXPRESSIONS)} expressions)")


# --------------------------------------------------------------------------
# Point-in-time records
# --------------------------------------------------------------------------

PIT_SYMBOLS = ("AAA", "BBB")
PIT_EXPRESSIONS = (
    "P($$roe_q)",
    "P($$eps_a)",
    "P(Mean($$roe_q, 4))",
    "P(Ref($$roe_q, 1))",
    "P(Sum($$ni_q, 4))",
    "P($$roe_q / $$ni_q)",
    "P(Max($$eps_a, 3))",
    "PRef($$roe_q, 201902)",
    "PRef($$eps_a, 2018)",
    "P(Delta($$roe_q, 1))",
    "P(Delta($$roe_q, 2))",
    "P(EMA($$roe_q, 4))",
    "P(Std($$ni_q, 4))",
    "P(IdxMax($$ni_q, 4))",
    "P(WMA($$roe_q, 3))",
    "P(Mean($$roe_q, 0))",
    "P($$roe_q) * 2 + 1",
)
HOLIDAYS = [
    *pd.date_range("2019-02-04", "2019-02-08"),
    *pd.date_range("2019-10-01", "2019-10-07"),
    *pd.date_range("2020-01-24", "2020-01-31"),
    *pd.date_range("2020-10-01", "2020-10-08"),
]

# (date, period, value) in the order they are published. Dates on weekends
# and holidays are visible on the next trading day.
AAA_ROE = [
    ("2017-04-27", 201701, 0.031),
    ("2017-08-28", 201702, 0.064),
    ("2017-10-27", 201703, 0.092),
    ("2018-03-28", 201704, 0.121),
    ("2018-04-26", 201801, 0.029),
    ("2018-08-27", 201802, 0.067),
    ("2018-10-29", 201803, 0.095),
    ("2019-03-30", 201804, 0.118),  # a Saturday
    ("2019-04-26", 201901, 0.033),
    ("2019-08-27", 201902, 0.071),
    ("2019-10-05", 201902, 0.069),  # a revision, during the October holiday
    ("2019-10-25", 201903, 0.101),
    # 201904 is never published: a gap inside the period list
    ("2020-04-28", 202001, 0.018),
    ("2020-04-28", 201903, 0.099),  # a revision on the same date as 202001
    ("2020-08-26", 202002, 0.052),
    ("2020-10-03", 202003, 0.083),  # a holiday
    ("2020-11-14", 202002, 0.055),  # a revision on a Saturday
]
AAA_NI_EXTRA = [
    ("2019-01-20", 201801, None),  # a Sunday revision of an old period
    ("2020-06-15", 202001, None),
]
BBB_ROE = [
    # the first record arrives after the calendar starts
    ("2019-03-20", 201804, 0.140),
    ("2019-04-29", 201901, 0.041),
    ("2019-08-30", 201902, 0.083),
    ("2019-09-02", 201901, 0.043),  # a correction of an earlier period
    ("2019-10-30", 201903, 0.122),
    ("2020-03-27", 201904, 0.151),
    ("2020-04-29", 202001, 0.026),
    ("2020-04-29", 201904, 0.149),  # same date as 202001
    ("2020-08-28", 202002, 0.061),
    ("2020-08-28", 202002, 0.063),  # same period, same date: the later record wins
    ("2020-10-29", 202003, 0.094),
]
BBB_NI_EXTRA = [("2019-12-07", 201902, None)]  # a Saturday
AAA_EPS = [
    ("2017-03-29", 2016, 1.21),
    ("2018-03-28", 2017, 1.35),
    ("2019-03-30", 2018, 1.52),  # a Saturday
    ("2019-06-10", 2018, 1.49),  # a revision
    ("2020-04-25", 2019, 1.18),  # a Saturday
]
BBB_EPS = [("2019-03-20", 2018, 0.87), ("2020-03-27", 2019, 0.95)]


def ni_records(roe: list, extra: list, scale: float, seed: float) -> list:
    """ni_q on roe_q's schedule (same periods on the same dates), plus extra
    revisions of already published periods."""
    out = [(date, period, round(scale * (1.0 + value * 7.0 + seed * (i % 3)), 4)) for i, (date, period, value) in enumerate(roe)]
    for k, (date, period, _) in enumerate(extra):
        out.append((date, period, round(scale * (2.0 + 0.37 * k + seed), 4)))
    return sorted(out, key=lambda r: r[0])  # stable: equal dates keep their order


PIT_RECORDS: dict[str, dict[str, list]] = {
    "AAA": {
        "roe_q": AAA_ROE,
        "ni_q": ni_records(AAA_ROE, AAA_NI_EXTRA, 1.0e3, 0.11),
        "eps_a": AAA_EPS,
    },
    "BBB": {
        "roe_q": BBB_ROE,
        "ni_q": ni_records(BBB_ROE, BBB_NI_EXTRA, 2.5e3, 0.07),
        "eps_a": BBB_EPS,
    },
}


def dump_pit(root: Path, symbol: str, field: str, records: list) -> list[tuple[int, int, float]]:
    """One field in ``scripts/dump_pit.py``'s layout: ``<field>.index`` is the
    first year then one record offset per period (NA where never
    published); ``<field>.data`` is ``(date, period, value, _next)`` in date
    order, each period's revisions linked through ``_next``. Values go
    through float32 as ``dump_pit.py`` casts them. Returns the records as
    written."""
    quarterly = field.endswith("_q")
    period_t, index_t = C.pit_record_type["period"], C.pit_record_type["index"]
    data_t = "".join(C.pit_record_type[k] for k in ("date", "period", "value", "index"))
    na_index = C.pit_record_nan["index"]
    rows = sorted(records, key=lambda r: r[0])  # stable sort by date
    periods = [p for _, p, _ in rows]
    first_year, last_year = min(periods), max(periods)
    if quarterly:
        first_year //= 100
        last_year //= 100
    slots = (last_year - first_year + 1) * (4 if quarterly else 1)
    index = [na_index] * slots
    tail: dict[int, int] = {}
    data = bytearray()
    written = []
    for date, period, value in rows:
        value = float(np.float32(value))
        date_int = int(date.replace("-", ""))
        offset = get_period_offset(first_year, period, quarterly)
        here = len(data)
        if index[offset] == na_index:
            index[offset] = here
        else:
            # Link from the period's last revision so far.
            prev = tail[offset]
            struct.pack_into(index_t, data, prev + struct.calcsize(data_t) - struct.calcsize(index_t), here)
        tail[offset] = here
        data += struct.pack(data_t, date_int, period, value, na_index)
        written.append((date_int, period, value))
    folder = root / "financial" / symbol.lower()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{field}.index").write_bytes(struct.pack(period_t, first_year) + struct.pack(index_t * slots, *index))
    (folder / f"{field}.data").write_bytes(bytes(data))
    return written


def read_back(root: Path, symbol: str, field: str) -> list[tuple[int, int, float]]:
    data_t = "".join(C.pit_record_type[k] for k in ("date", "period", "value", "index"))
    raw = (root / "financial" / symbol.lower() / f"{field}.data").read_bytes()
    return [struct.unpack_from(data_t, raw, at)[:3] for at in range(0, len(raw), struct.calcsize(data_t))]


def write_pit() -> None:
    calendar = pd.bdate_range("2019-01-02", "2020-12-31").difference(pd.DatetimeIndex(HOLIDAYS))
    rows = len(calendar)
    # A throwaway provider directory (set TMPDIR to choose where).
    root = Path(tempfile.mkdtemp(prefix="qlib_pit_"))
    (root / "calendars").mkdir()
    (root / "calendars" / "day.txt").write_text("".join(f"{d.date().isoformat()}\n" for d in calendar))
    (root / "instruments").mkdir()
    (root / "instruments" / "all.txt").write_text(
        "".join(f"{s}\t{calendar[0].date()}\t{calendar[-1].date()}\n" for s in PIT_SYMBOLS)
    )
    (root / "features").mkdir()

    written: list[tuple[str, str, int, int, float]] = []
    for symbol in PIT_SYMBOLS:
        for field, records in PIT_RECORDS[symbol].items():
            dumped = dump_pit(root, symbol, field, records)
            assert dumped == read_back(root, symbol, field)
            written += [(symbol, field, *record) for record in dumped]

    qlib.init(provider_uri=str(root), region="cn", clear_mem_cache=True)
    provider = LocalExpressionProvider(time2idx=False)
    output: dict[str, dict[str, np.ndarray]] = {symbol: {} for symbol in PIT_SYMBOLS}
    for expression in PIT_EXPRESSIONS:
        H["f"].clear()
        for symbol in PIT_SYMBOLS:
            values = provider.get_expression_instance(expression).load(symbol, 0, rows - 1, "day")
            values = values.reindex(pd.RangeIndex(0, rows))
            output[symbol][expression] = values.to_numpy(dtype=np.float64)

    out = FIXTURES / "qlib_pit_records.tsv"
    with out.open("w", newline="") as handle:
        versions(handle)
        handle.write("# records in the order written to Qlib's <field>.data files (date order, stable)\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["symbol", "field", "date", "period", "value"])
        for symbol, field, date_int, period, value in written:
            date = f"{date_int // 10000:04d}-{date_int // 100 % 100:02d}-{date_int % 100:02d}"
            writer.writerow([symbol, field, date, period, format_float(value)])
    print(f"wrote {out} ({len(written)} records)")

    out = FIXTURES / "qlib_pit.tsv"
    with out.open("w", newline="") as handle:
        versions(handle)
        handle.write(f"# layout=symbol-major rows={rows} symbols={','.join(PIT_SYMBOLS)}\n")
        handle.write("# P() is float32 in Qlib; values are the float64 of that float32\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["symbol", "row", "date", *PIT_EXPRESSIONS])
        for symbol in PIT_SYMBOLS:
            for row, date in enumerate(calendar):
                writer.writerow(
                    [
                        symbol,
                        row,
                        date.date().isoformat(),
                        *(format_float(float(output[symbol][e][row])) for e in PIT_EXPRESSIONS),
                    ]
                )
    print(f"wrote {out} ({len(PIT_SYMBOLS) * rows} rows, {len(PIT_EXPRESSIONS)} expressions)")


# --------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------

SEM_SYMBOLS = ("AAA", "BBB")
SEM_ROWS = 60
SEM_INPUTS = ("close", "volume", "收盘价", "成交量（手）")
SEM_EXPRESSIONS = [
    "Ref($close, 0)",
    "Delta($close, 0)",
    "$close/Ref($close, 0) - 1",
    "Mean($close, 0.5)",
    "Std($close, 0.5)",
    "Var($close, 0.5)",
    "Max($close, 0.3)",
    "Min($close, 0.3)",
    "Sum($volume, 0.25)",
    "Med($close, 0.5)",
    "Count($close, 0.5)",
    "EMA($close, 0.5)",
    "Mean($close, 2*5)",
    "Std($close, 3*4-2)",
    "Ref($close, 4-2)",
    "Corr($close, $volume, 2*5)",
    "If($close > Ref($close, 1), True, False)",
    "$收盘价/$close",
    "Mean($成交量（手）, 5)",
]


def sem_panel() -> dict[str, dict[str, np.ndarray]]:
    data = {}
    for k, symbol in enumerate(SEM_SYMBOLS):
        row = np.arange(SEM_ROWS, dtype=np.float64)
        close = 20.0 + k * 7.0 + np.sin(row * 0.29 + k) * 2.0 + row * 0.05
        volume = 1000.0 + ((row * 37 + k * 11) % 23) * 40.0
        fields = {
            "close": close,
            "volume": volume,
            "收盘价": close * 1.01 + 0.3,
            "成交量（手）": volume / 100.0,
        }
        # Missing rows, and for BBB a missing first row, which Ref(x, 0) reads.
        fields["close"][[9 + k, 31, 44 - k]] = np.nan
        fields["volume"][[17, 50]] = np.nan
        if symbol == "BBB":
            fields["close"][0] = np.nan
        data[symbol] = fields
    return data


def write_semantics() -> None:
    data = sem_panel()
    dates = pd.bdate_range("2021-03-01", periods=SEM_ROWS)

    def load_feature(self: Feature, instrument: str, start: int, end: int, *_args) -> pd.Series:
        return pd.Series(data[instrument][self._name][start : end + 1], index=dates[start : end + 1])

    Feature._load_internal = load_feature
    provider = LocalExpressionProvider(time2idx=False)
    output: dict[str, dict[str, np.ndarray]] = {symbol: {} for symbol in SEM_SYMBOLS}
    for expression in SEM_EXPRESSIONS:
        H["f"].clear()
        for symbol in SEM_SYMBOLS:
            values = provider.get_expression_instance(expression).load(symbol, 0, SEM_ROWS - 1, "day")
            output[symbol][expression] = values.to_numpy(dtype=np.float64)

    out = FIXTURES / "qlib_semantics.tsv"
    with out.open("w", newline="") as handle:
        versions(handle)
        handle.write(f"# layout=symbol-major rows={SEM_ROWS} symbols={','.join(SEM_SYMBOLS)}; booleans are 1/0\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["symbol", "row", "date", *SEM_INPUTS, *SEM_EXPRESSIONS])
        for symbol in SEM_SYMBOLS:
            for row, date in enumerate(dates):
                writer.writerow(
                    [
                        symbol,
                        row,
                        date.date().isoformat(),
                        *(format_float(float(data[symbol][name][row])) for name in SEM_INPUTS),
                        *(format_float(float(output[symbol][e][row])) for e in SEM_EXPRESSIONS),
                    ]
                )
    print(f"wrote {out} ({len(SEM_SYMBOLS) * SEM_ROWS} rows, {len(SEM_EXPRESSIONS)} expressions)")


def main() -> None:
    qlib.init(custom_ops=[DayCumsum, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut])
    write_alpha158_configs()
    write_highfreq()
    write_semantics()
    write_pit()


if __name__ == "__main__":
    main()
