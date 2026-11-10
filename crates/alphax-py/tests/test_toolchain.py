"""Config files, the CLI, scheduling, persistence and developer tools."""

import datetime as dt
import json

import numpy as np
import pytest

import alphax
from alphax import cli, config, schedule


def _bars(tmp_path, n=200, cols=3):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, cols)), axis=0))
    np.savez(tmp_path / "bars.npz", close=close, high=close * 1.01, low=close * 0.99,
             volume=rng.uniform(1e3, 2e3, (n, cols)))
    return close


CONFIG = """
[data]
path = "bars.npz"

[signals]
entries = "$close > Mean($close, {window})"
exits = "$close < Mean($close, {window})"

[backtest]
size = 1.0
fees = 0.001
init_cash = 10000.0

[sweep]
window = [5, 20]

[output]
stats = "stats.json"
portfolio = "last.alphax"
"""


def test_a_config_runs_like_the_python_it_describes(tmp_path):
    close = _bars(tmp_path)
    (tmp_path / "bt.toml").write_text(CONFIG)
    result = config.run(tmp_path / "bt.toml")
    assert [p["window"] for p in result["params"]] == [5, 20]
    # Qlib's Mean: partial windows from the first bar.
    csum = np.cumsum(close, axis=0)
    partial = np.vstack([csum[:20] / np.arange(1, 21)[:, None], (csum[20:] - csum[:-20]) / 20])
    # With warm-up (the default) no signal before the 20-bar mean has filled.
    ma = partial.copy()
    ma[:19] = np.nan
    market = dict(high=close * 1.01, low=close * 0.99, size=1.0, fees=0.001, init_cash=10000.0)
    direct = alphax.from_signals(close, close > ma, close < ma, **market)
    assert len(direct.orders()["idx"]) > 10
    np.testing.assert_allclose(result["portfolios"][1].value(), direct.value())
    saved = json.loads((tmp_path / "stats.json").read_text())
    assert saved[1]["params"] == {"window": 20} and len(saved[1]["stats"]) == 3
    np.testing.assert_array_equal(alphax.load(tmp_path / "last.alphax").value(), result["portfolios"][1].value())
    # warmup = false trades on the partial windows.
    (tmp_path / "raw.toml").write_text(CONFIG.replace("[backtest]\n", "[backtest]\nwarmup = false\n"))
    raw = config.run(tmp_path / "raw.toml")
    qlib = alphax.from_signals(close, close > partial, close < partial, **market)
    assert qlib.orders()["idx"][0] < 19 <= direct.orders()["idx"][0]
    np.testing.assert_allclose(raw["portfolios"][1].value(), qlib.value())


def test_configs_are_checked(tmp_path):
    (tmp_path / "a.toml").write_text('[data]\npath="x.npz"\n[signals]\nentries="$close > {w}"\nexits="$close < 0"\n')
    with pytest.raises(ValueError, match="placeholder"):
        config.load(tmp_path / "a.toml")
    (tmp_path / "b.toml").write_text('[data]\npath="x.npz"\n[signals]\nentries="1"\n[extra]\n')
    with pytest.raises(ValueError, match="unknown section"):
        config.load(tmp_path / "b.toml")
    (tmp_path / "c.toml").write_text('[data]\npath="x.npz"\n[signals]\nentries="1"\n[backtest]\nentry="from_orders"\n')
    with pytest.raises(ValueError, match="needs size"):
        config.load(tmp_path / "c.toml")
    (tmp_path / "d.toml").write_text('[data]\npath="x.npz"\n[signals]\nentries="1"\nexits="0"\n[backtest]\nwarmup="no"\n')
    with pytest.raises(ValueError, match="warmup"):
        config.load(tmp_path / "d.toml")


def test_a_nan_signal_is_no_signal():
    values = np.array([[np.nan, 0.0, 1.0, -2.0, np.inf]])
    np.testing.assert_array_equal(config._signal(values), [[False, False, True, True, True]])


def test_long_csv_data(tmp_path):
    rows = ["time,symbol,close,high,low"]
    for t in range(30):
        for s, base in (("A", 10.0), ("B", 20.0)):
            rows.append(f"2024-01-{t + 1:02d},{s},{base + t},{base + t + 1},{base + t - 1}")
    (tmp_path / "bars.csv").write_text("\n".join(rows))
    data = config.load_data({"csv": "bars.csv"}, tmp_path)
    assert data["close"].shape == (30, 2) and data["close"][3, 1] == 23.0
    assert list(data["symbols"]) == ["A", "B"]


def test_cli_run_sweep_schedule_and_help(tmp_path, capsys):
    _bars(tmp_path)
    (tmp_path / "bt.toml").write_text(CONFIG)
    log = tmp_path / "runs.log"
    (tmp_path / "notify.toml").write_text(CONFIG + '\n[notify]\nfile = "from_config.log"\n')
    cli.main(["run", str(tmp_path / "notify.toml")])
    assert "done" in (tmp_path / "from_config.log").read_text()
    capsys.readouterr()
    assert cli.main(["run", str(tmp_path / "bt.toml"), "--notify-file", str(log)]) == 0
    out = capsys.readouterr().out
    assert "window=5" in out and "window=20" in out
    assert "done" in log.read_text() and "account 0: total return" in log.read_text()
    assert "None" not in out
    cli.main(["sweep", str(tmp_path / "bt.toml"), "--set", "window=3,7,9", "--metric", "sharpe_ratio"])
    out = capsys.readouterr().out
    assert out.count("window=") == 3 and out.startswith("Sharpe Ratio per account")
    cli.main(["schedule", str(tmp_path / "bt.toml"), "--at", "07:05", "--weekdays", "1-5"])
    line = capsys.readouterr().out
    assert line.startswith("5 7 * * 1-5 ") and "alphax.cli run" in line
    cli.main(["help", "indicators.rsi"])
    assert "rsi(" in capsys.readouterr().out


def test_jobs_report_failures_and_schedules_compute_times(tmp_path):
    log = tmp_path / "log"
    sent = []
    job = schedule.Job(lambda: 1 / 0, name="bad", notify=[schedule.File(str(log)), lambda s, t: sent.append(s)])
    with pytest.raises(ZeroDivisionError):
        job()
    assert sent == ["bad failed"] and "ZeroDivisionError" in log.read_text()
    now = dt.datetime(2024, 5, 1, 7, 0)
    assert schedule.next_time(now, at="06:30") == dt.datetime(2024, 5, 2, 6, 30)
    assert schedule.next_time(now, at="08:00") == dt.datetime(2024, 5, 1, 8, 0)
    runs, waits = [], []
    schedule.every(lambda: runs.append(1), interval=60, times=3, sleep=waits.append,
                   clock=lambda: now)
    assert len(runs) == 3 and waits == [60.0] * 3


def test_save_and_load_research_objects(tmp_path):
    close = np.random.default_rng(0).uniform(90, 110, (50, 2))
    pf = alphax.from_signals(close, close > 100, close < 100, size=1.0)
    res = alphax.params.run(lambda w: {"pf": pf, "w": w}, alphax.params.grid(w=[1, 2]))
    alphax.save(res, tmp_path / "res.bin")
    back = alphax.load(tmp_path / "res.bin")
    assert back.params == res.params
    np.testing.assert_array_equal(back.values[1]["pf"].value(), pf.value())
    alphax.save({"a": np.arange(3), "pfs": [pf]}, tmp_path / "d.bin")
    d = alphax.load(tmp_path / "d.bin")
    np.testing.assert_array_equal(d["a"], np.arange(3))
    assert isinstance(d["pfs"][0], alphax.Portfolio)
    alphax.save(pf, tmp_path / "pf.bin")
    assert isinstance(alphax.load(tmp_path / "pf.bin"), alphax.Portfolio)


def test_check_shapes_and_help():
    close = np.ones((10, 3))
    out = alphax.check_shapes(close, fees=0.001, size=np.ones(3), cash_deposits=np.zeros(10),
                              high=np.ones((10, 3)), sl_stop=None)
    assert out == {"fees": "scalar", "size": "per column", "cash_deposits": "per bar",
                   "high": "full", "sl_stop": "default"}
    with pytest.raises(ValueError, match="1x3 array against a 10x3"):
        alphax.check_shapes(close, fees=np.ones((1, 3)))
    with pytest.raises(ValueError, match="one value per column"):
        alphax.check_shapes(close, size=np.ones(10))
    with pytest.raises(ValueError, match="funding_rate"):
        alphax.check_shapes(close, funding_rate=np.ones(3))
    text = alphax.help("from_signals")
    assert text.startswith("from_signals(")
    assert "members:" in alphax.help(alphax.allocation)
