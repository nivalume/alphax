"""Qlib parity through the Python API: configurable Alpha158, the label, the
high-frequency operators, ChangeInstrument and point-in-time P / PRef.

The fixtures are Qlib's own outputs (tests/generators/generate_qlib_extensions.py).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import alphax
import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"


def table(name: str) -> tuple[list[str], list[list[str]], list[str]]:
    lines = (FIXTURES / name).read_text().splitlines()
    comments = [line for line in lines if line.startswith("#")]
    rows = list(csv.reader((line for line in lines if not line.startswith("#")), delimiter="\t"))
    return rows[0], rows[1:], comments


def check(actual: np.ndarray, expected: np.ndarray, epsilon: float, label: str) -> None:
    assert actual.shape == expected.shape, label
    nan = np.isnan(expected)
    assert np.array_equal(np.isnan(actual), nan), label
    finite = np.isfinite(expected)
    assert np.array_equal(actual[~finite & ~nan], expected[~finite & ~nan]), label
    tolerance = epsilon * np.maximum(np.abs(expected[finite]), 1.0)
    assert np.all(np.abs(actual[finite] - expected[finite]) <= tolerance), label


def test_alpha158_configs_match_qlib():
    header, rows, comments = table("qlib_alpha158_configs.tsv")
    configs = {}
    for line in comments:
        if line.startswith("# config "):
            name, _, spec = line[len("# config "):].partition(" = ")
            configs[name] = None if spec == "qlib default" else json.loads(spec)
    assert len(configs) == 6
    for name, config in configs.items():
        expected = [(row[2], row[3]) for row in rows if row[0] == name]
        assert alphax.alpha158_features(config) == expected, name
        assert alphax.alpha158(config).names == tuple(n for n, _ in expected), name
    assert len(alphax.alpha158().names) == 158


def test_alpha158_config_errors():
    with pytest.raises(ValueError, match="LOW"):
        alphax.alpha158({"rolling": {"include": ["MIN"]}})
    with pytest.raises(ValueError, match="sections"):
        alphax.alpha158({"kbar": {}, "labels": {}})
    with pytest.raises(ValueError, match="empty"):
        alphax.alpha158({})


def test_qlib_label():
    close = np.array([[10.0], [11.0], [12.1], [12.1], [13.0]])
    label = alphax.qlib_label()
    assert label.uses_future and label.names == ("LABEL0",)
    values = label.evaluate({"close": close})["LABEL0"][:, 0]
    np.testing.assert_allclose(values[:3], [0.1, 0.0, 13.0 / 12.1 - 1])
    assert np.isnan(values[3:]).all()
    assert alphax.qlib_label("vwap").required_fields == ("vwap",)


def highfreq():
    header, rows, _ = table("qlib_highfreq.tsv")
    symbols = list(dict.fromkeys(row[0] for row in rows))
    n_rows = len(rows) // len(symbols)

    def panel(name: str) -> np.ndarray:
        at = header.index(name)
        return np.array([float(row[at]) for row in rows]).reshape(len(symbols), n_rows).T

    index = np.array([row[2] for row in rows[:n_rows]], dtype="datetime64[ns]")
    fields = {name: panel(name) for name in ("close", "volume", "paused_num", "x")}
    expected = {name: panel(name) for name in header[7:]}
    return fields, index, symbols, expected


def test_high_frequency_and_instrument_operators_match_qlib():
    fields, index, symbols, expected = highfreq()
    factors = alphax.FactorSet.compile({e: e for e in expected})
    assert factors.needs_timestamps
    assert factors.instrument_names == ("BBB",)
    values = factors.evaluate(fields, index=index, symbols=symbols)
    for name, want in expected.items():
        check(values[name], want, 1e-6 if name.startswith("Corr") else 1e-9, name)

    batches = {}

    def sink(start, stop, part):
        for name, v in part.items():
            batches.setdefault(name, []).append(v)

    factors.evaluate_batches(fields, 1, sink, index=index, symbols=symbols)
    for name, v in values.items():
        np.testing.assert_array_equal(np.hstack(batches[name]), v, err_msg=name)


def test_timezone_aware_index_reads_the_market_clock():
    pd = pytest.importorskip("pandas")
    fields, index, symbols, expected = highfreq()
    aware = pd.DatetimeIndex(index).tz_localize("Asia/Shanghai")
    name = "DayCumsum($volume, '10:15', '13:44', 30)"
    values = alphax.FactorSet.compile({name: name}).evaluate(fields, index=aware, symbols=symbols)
    check(values[name], expected[name], 1e-9, name)


def test_an_instrument_outside_the_panel():
    fields, index, symbols, expected = highfreq()
    name = "$close/ChangeInstrument('BBB', $close)"
    factors = alphax.FactorSet.compile({name: name})
    alone = factors.evaluate(
        {"close": fields["close"][:, :1]},
        index=index,
        instruments={"BBB": {"close": fields["close"][:, 1]}},
    )
    check(alone[name], expected[name][:, :1], 1e-9, name)
    with pytest.raises(ValueError, match="BBB"):
        factors.evaluate({"close": fields["close"][:, :1]}, index=index)


def pit_inputs():
    _, records, _ = table("qlib_pit_records.tsv")
    header, rows, _ = table("qlib_pit.tsv")
    symbols = list(dict.fromkeys(row[0] for row in rows))
    n_rows = len(rows) // len(symbols)
    pit = {
        "symbol": [r[0] for r in records],
        "field": [r[1] for r in records],
        "date": [r[2] for r in records],
        "period": [int(r[3]) for r in records],
        "value": [float(r[4]) for r in records],
    }
    index = np.array([row[2] for row in rows[:n_rows]], dtype="datetime64[D]")
    expected = {
        name: np.array([float(row[3 + k]) for row in rows]).reshape(len(symbols), n_rows).T
        for k, name in enumerate(header[3:])
    }
    return pit, index, symbols, expected


def test_point_in_time_operators_match_qlib():
    pit, index, symbols, expected = pit_inputs()
    factors = alphax.FactorSet.compile({e: e for e in expected})
    assert not factors.uses_future
    values = factors.evaluate({}, index=index, symbols=symbols, pit=pit)
    for name, want in expected.items():
        check(values[name], want, 1e-6, name)

    # Qlib's YYYYMMDD dates, a DataFrame, and symbols of other panels dropped.
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(pit)
    frame["date"] = frame["date"].str.replace("-", "").astype(int)
    extra = frame.iloc[:3].assign(symbol="ZZZ", value=1e9)
    again = factors.evaluate(
        {}, index=index, symbols=symbols, pit=pd.concat([frame, extra], ignore_index=True)
    )
    for name in expected:
        np.testing.assert_array_equal(again[name], values[name], err_msg=name)


def test_to_parquet_passes_the_index_and_symbols(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pit, index, symbols, expected = pit_inputs()
    factors = alphax.FactorSet.compile({"roe": "P($$roe_q)"})
    path = tmp_path / "pit.parquet"
    factors.to_parquet(str(path), {}, batch=1, index=index, symbols=symbols, pit=pit)
    got = pq.read_table(path).column("roe").to_numpy().reshape(len(symbols), len(index)).T
    check(got, expected["P($$roe_q)"], 1e-6, "roe")
