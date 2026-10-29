"""Python market-data boundary."""

import runpy
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT / "examples"))
import binance as data
import phasor

DAY = 86_400_000
ROOT = Path(__file__).parents[3]


def row(open_time, close=100.0):
    return [
        open_time,
        str(close - 1.0),
        str(close + 2.0),
        str(close - 3.0),
        str(close),
        "42.5",
        open_time + DAY - 1,
        "0",
        7,
        "0",
        "0",
        "0",
    ]


class Fake:
    def __init__(self, pages, now=10 * DAY):
        self.pages = list(pages)
        self.now = now
        self.urls = []
        self.paced = []

    def get(self, url):
        self.urls.append(url)
        return self.pages.pop(0)

    def server_time(self):
        return self.now

    def pace(self, seconds):
        self.paced.append(seconds)


def starts(fake):
    return [int(parse_qs(urlparse(url).query)["startTime"][0]) for url in fake.urls]


def test_fetch_pages_forward_and_drops_the_open_bar():
    fake = Fake([[row(0), row(DAY)], [row(2 * DAY)]], now=2 * DAY + DAY // 2)
    bars = data.fetch_klines("BTCUSDT", "1d", 0, limit=2, pace=0.1, transport=fake)

    assert [bar.open_time for bar in bars] == [0, DAY]
    assert starts(fake) == [0, DAY + 1]
    assert fake.paced == [0.1]


def test_fetch_stops_at_the_request_ceiling():
    fake = Fake([[row(0)], [row(DAY)], [row(2 * DAY)]])
    with pytest.raises(ValueError, match="stopped after 2 requests"):
        data.fetch_klines(
            "BTCUSDT",
            "1d",
            0,
            limit=1,
            max_requests=2,
            closed_only=False,
            transport=fake,
        )


def test_cache_round_trips_and_resume_starts_after_the_last_bar(tmp_path):
    original = 29_000.123_456_789_1
    first = Fake([[row(0, original), row(DAY, 0.000_000_012_3)]])
    bars = data.fetch_cached("BTCUSDT.BINANCE", "1d", 0, directory=tmp_path, transport=first)
    loaded = data.load_cached("BTCUSDT.BINANCE", "1d", directory=tmp_path)

    assert [bar.close for bar in loaded] == [bar.close for bar in bars]
    assert loaded[0].close.hex() == original.hex()

    second = Fake([[]])
    assert data.fetch_cached(
        "BTCUSDT.BINANCE", "1d", 0, directory=tmp_path, transport=second
    ) == loaded
    assert starts(second) == [2 * DAY]


@pytest.mark.parametrize(
    "contents,message",
    [
        ("time,price\n1,2\n", "header"),
        (
            "open_time,open,high,low,close,volume,close_time\n"
            "86400000,1,1,1,1,1,2\n0,1,1,1,1,1,2\n",
            "does not follow",
        ),
        (
            "open_time,open,high,low,close,volume,close_time\n1,x,3,4,5,6,7\n",
            "non-numeric",
        ),
    ],
)
def test_bad_cache_is_rejected(tmp_path, contents, message):
    (tmp_path / "BTCUSDT.BINANCE-1d.csv").write_text(contents)
    with pytest.raises(ValueError, match=message):
        data.load_cached("BTCUSDT.BINANCE", "1d", directory=tmp_path)


def test_sweep_example_runs(capsys):
    runpy.run_path(ROOT / "examples/sweep.py", run_name="__main__")
    assert "best:" in capsys.readouterr().out


def test_factor_evaluation_example_runs(capsys):
    runpy.run_path(ROOT / "examples/factor_evaluation.py", run_name="__main__")
    out = capsys.readouterr().out
    assert "momentum" in out
    assert "p_value" in out
    assert "\nok" in out


def test_fetch_example_runs_through_fake_transport(tmp_path, monkeypatch, capsys):
    fake = Fake([[row(0), row(DAY)]])
    fetch_cached = data.fetch_cached

    def fetch(symbol, interval, start, **kwargs):
        return fetch_cached(symbol, interval, start, transport=fake, **kwargs)

    monkeypatch.setattr(data, "fetch_cached", fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch.py",
            "--symbols",
            "BTCUSDT.BINANCE",
            "--interval",
            "1d",
            "--from",
            "1970-01-01",
            "--dir",
            str(tmp_path),
        ],
    )
    runpy.run_path(ROOT / "examples/fetch.py", run_name="__main__")

    assert "2 bars" in capsys.readouterr().out
    assert len(data.load_cached("BTCUSDT.BINANCE", "1d", directory=tmp_path)) == 2


def test_crypto_example_runs_from_cache(tmp_path, monkeypatch, capsys):
    prices = [row(index * DAY, 100.0 + 10.0 * np.sin(index / 10.0)) for index in range(250)]
    fake = Fake([prices, prices], now=300 * DAY)
    for symbol in ("BTCUSDT.BINANCE", "ETHUSDT.BINANCE"):
        data.fetch_cached(symbol, "1d", 0, directory=tmp_path, transport=fake)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto.py",
            "--symbols",
            "BTCUSDT.BINANCE,ETHUSDT.BINANCE",
            "--interval",
            "1d",
            "--dir",
            str(tmp_path),
        ],
    )
    runpy.run_path(ROOT / "examples/crypto.py", run_name="__main__")
    assert "coverage=" in capsys.readouterr().out


def test_crypto_demo_runs_offline(monkeypatch, capsys):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(sys, "argv", ["crypto.py", "--demo"])
    runpy.run_path(ROOT / "examples/crypto.py", run_name="__main__")
    assert "coverage=" in capsys.readouterr().out
