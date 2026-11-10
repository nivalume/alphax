"""Indicators, signals, labels, return-series analysis, portfolio transforms,
pattern search, frames and plots through the Python API."""

import numpy as np
import pytest

import alphax


def _close(n=120, cols=2, seed=0):
    rng = np.random.default_rng(seed)
    return np.asfortranarray(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, cols)), axis=0)))


def _pf(seed=0, **kw):
    rng = np.random.default_rng(seed)
    c = _close(seed=seed, cols=3)
    e = rng.random(c.shape) < 0.08
    x = rng.random(c.shape) < 0.08
    return alphax.from_signals(c, e, x, size=1.0, direction="both", fees=0.001, **kw), c


# --- indicators -------------------------------------------------------------


def test_indicators_sweep_and_label_their_columns():
    c = _close()
    r = alphax.indicators.ma(c, window=[5, 10], ewm=[False, True])
    assert r.ma.shape == (len(c), 8)
    assert r.columns["window"] == [5, 5, 5, 5, 10, 10, 10, 10]
    assert r.columns["input"] == [0, 1] * 4
    sma10 = r.select(window=10, ewm=False).ma
    np.testing.assert_allclose(sma10[-1], c[-10:].mean(axis=0))
    assert np.isnan(sma10[8]).all() and not np.isnan(sma10[9]).any()
    paired = alphax.indicators.ma(c, window=[5, 10], ewm=[False, True], product=False)
    assert paired.params()[0] == {"window": 5, "ewm": False, "input": 0}
    assert len(paired.params()) == 4
    for f, args in [
        (alphax.indicators.mstd, (c,)),
        (alphax.indicators.bbands, (c,)),
        (alphax.indicators.rsi, (c,)),
        (alphax.indicators.macd, (c,)),
        (alphax.indicators.stoch, (c * 1.01, c * 0.99, c)),
        (alphax.indicators.atr, (c * 1.01, c * 0.99, c)),
        (alphax.indicators.obv, (c, np.ones_like(c))),
    ]:
        out = f(*args)
        assert all(v.shape[0] == len(c) for v in out.outputs.values())
    assert alphax.indicators.true_range(c * 1.01, c * 0.99, c).shape == c.shape
    with pytest.raises(AttributeError, match="no output"):
        r.nope


def test_indicator_factory():
    c = _close()

    def momentum(close, window):
        out = np.full_like(close, np.nan)
        out[window:] = close[window:] / close[:-window] - 1
        return out

    f = alphax.indicators.IndicatorFactory(momentum, inputs=["close"], params=["window"], outputs=["mom"])
    r = f.run(c, window=[3, 5])
    assert r.mom.shape == (len(c), 4)
    np.testing.assert_allclose(r.select(window=5, input=1).mom[:, 0], momentum(c[:, 1], 5))
    with pytest.raises(TypeError, match="unknown"):
        f.run(c, span=3)
    with pytest.raises(TypeError, match="no value"):
        f.run(c)


# --- signals and labels ------------------------------------------------------


def test_signal_tools():
    s = alphax.signals
    e = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], bool)
    x = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0], bool)
    ce, cx = s.clean(e, x)
    np.testing.assert_array_equal(np.flatnonzero(ce), [0, 4, 6, 9])
    np.testing.assert_array_equal(np.flatnonzero(cx), [2, 5, 8])
    # The entry on bar 4 has no bar 2 later before the next entry, on 6.
    np.testing.assert_array_equal(np.flatnonzero(s.exit_after(ce, 2)), [2, 8])
    r = s.ranges(e)
    np.testing.assert_array_equal(r["start_idx"], [0, 4, 6, 9])
    np.testing.assert_array_equal(r["status"], [1, 1, 1, 0])
    np.testing.assert_array_equal(s.rank(e)[:2], [0, 1])
    np.testing.assert_array_equal(s.rank(e, partition=True)[[0, 4, 6]], [0, 1, 2])
    assert s.nth_index(e, 1) == 1 and s.nth_index(e, -1) == 9
    cond = np.zeros(10, bool)
    cond[7] = True
    np.testing.assert_array_equal(np.flatnonzero(s.exit_on(ce, cond, until="end")), [7])
    st = s.stats(e)
    assert st["count"][0] == 5 and st["first_index"][0] == 0
    rl = s.random_like(np.column_stack([e, e]), keep="count", seed=3)
    np.testing.assert_array_equal(rl.sum(axis=0), [5, 5])
    rg = s.random_like(e, keep="gaps", seed=1)
    assert sorted(np.diff(np.flatnonzero(rg))) == sorted(np.diff(np.flatnonzero(e)))


def test_calendar_signals():
    days = np.arange("2024-01-01", "2024-04-30", dtype="datetime64[D]")
    weekdays = days[(days.astype(np.int64) + 3) % 7 < 5]
    first = alphax.signals.calendar(weekdays, "ME")
    np.testing.assert_array_equal(weekdays[first].astype(str),
                                  ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    last = alphax.signals.calendar(weekdays, "ME", which="last")
    assert weekdays[last][1] == np.datetime64("2024-02-29")
    third = alphax.signals.calendar(weekdays, "W", n=2)
    assert all(((d.astype(np.int64) + 3) % 7) == 2 for d in weekdays[third][1:])
    with pytest.raises(ValueError, match="freq"):
        alphax.signals.calendar(weekdays, "H")


def test_mean_labels_and_predictions():
    c = _close()
    lab = alphax.labels.mean_labels(c, 3)
    np.testing.assert_allclose(lab[0], c[1:4].mean(axis=0) / c[0] - 1)
    assert np.isnan(lab[-3:]).all()
    pred = np.vstack([np.diff(c, axis=0), np.full((1, 2), np.nan)])  # a look-ahead "model"
    res = alphax.predictions.backtest(c, pred)
    assert set(res) == {"model", "hold", "random"}
    table = alphax.predictions.compare(res)
    # Trading on tomorrow's move beats holding and chance.
    assert (table["total_return"]["model"] > table["total_return"]["hold"]).all()
    assert (table["total_return"]["model"] > table["total_return"]["random"]).all()


# --- returns analysis --------------------------------------------------------


def test_return_series_match_the_portfolio():
    pf, c = _pf()
    r = np.column_stack([pf.returns(a) for a in range(3)])
    np.testing.assert_allclose(alphax.returns.metric(r, "sharpe_ratio"), pf.sharpe_ratio())
    np.testing.assert_allclose(alphax.returns.metric(r, "max_drawdown"), pf.max_drawdown())
    np.testing.assert_array_equal(pf.rolling("sharpe_ratio", 30), alphax.returns.rolling(r, "sharpe_ratio", 30))
    st = alphax.returns.stats(r[:, 0])
    assert st["drawdown_count"] >= 1 and st["total_return"] == pytest.approx(pf.total_return()[0])
    with pytest.raises(ValueError, match="needs a benchmark"):
        alphax.returns.metric(r, "beta")
    with pytest.raises(ValueError, match="no metric"):
        alphax.returns.metric(r, "magic")
    assert pf.rolling("beta", 30).shape == r.shape
    d = pf.drawdowns()
    assert {"peak_idx", "valley_idx", "end_idx", "drawdown", "recovered"} <= set(d)
    assert (d["drawdown"] <= 0).all()
    s = pf.stats(0, high=c * 1.01, low=c * 0.99)
    assert {"avg_mae_long", "avg_mfe_short", "edge_ratio_long"} <= set(s)
    assert "avg_mae_long" not in pf.stats(0)


# --- portfolio transforms ----------------------------------------------------


def test_portfolio_transforms_reproduce_the_original():
    P = alphax.portfolios
    pf, c = _pf(init_cash=[100.0, 200.0, 300.0])
    back = alphax.Portfolio.from_records(pf.orders(), pf.close, init_cash=pf.init_cash,
                                         cash_records=pf.cash_records())
    np.testing.assert_array_equal(back.value(1), pf.value(1))
    sel = P.select(pf, [2, 0])
    np.testing.assert_array_equal(sel.value(0), pf.value(2))
    np.testing.assert_array_equal(P.column_stack(pf, sel).value(4), pf.value(0))
    cut = P.row_slice(pf, 40, 100)
    np.testing.assert_allclose(cut.value(1), pf.value(1)[40:100])
    np.testing.assert_allclose(cut.returns(1), pf.returns(1)[40:100])
    bars = P.resample_bars(np.arange("2024-01-01", len(c), dtype="datetime64[D]"), "W")
    last = np.flatnonzero(np.diff(np.append(bars, bars[-1] + 1)))
    np.testing.assert_allclose(P.resample(pf, bars).value(0), pf.value(0)[last])
    shared = alphax.from_signals(c, c > 100, c < 99, size=1.0, group_lens=[2, 1], cash_sharing=True)
    np.testing.assert_allclose(P.row_slice(shared, 30).value(group=0), shared.value(group=0)[30:])
    np.testing.assert_array_equal(P.select(shared, [1]).value(group=0), shared.value(group=1))
    with pytest.raises(NotImplementedError):
        P.row_slice(alphax.from_signals(c, c > 100, c < 99, size=1.0, leverage=2.0), 10)
    with pytest.raises(ValueError, match="column 7"):
        alphax.Portfolio.from_records({"col": [7], "idx": [0], "size": [1.0], "price": [1.0],
                                       "fees": [0.0], "side": [0]}, pf.close)


def test_pattern_search():
    x = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 0, 10), np.zeros(10)]) * 5 + 100
    m = alphax.patterns.search(x, [0, 1, 0], window=[20], min_similarity=0.95)
    assert list(m["start_idx"]) == [0] and m["similarity"][0] > 0.99
    sig = alphax.patterns.signals(m, (len(x), 1))
    assert sig[20, 0] and sig.sum() == 1


def test_frames_and_plots():
    pd = pytest.importorskip("pandas")
    pf, _ = _pf()
    idx = np.arange("2024-01-01", 120, dtype="datetime64[D]")
    df = alphax.frames.orders(pf, index=idx)
    assert isinstance(df, pd.DataFrame) and (df["time"].to_numpy() == idx[df["idx"]]).all()
    assert alphax.frames.stats(pf).shape[1] == 3
    pytest.importorskip("plotly")
    assert len(alphax.plotting.value(pf, 0, index=idx).data) == 2
    assert len(alphax.plotting.orders(pf, 1).data) == 3


def test_talib_wrapper_matches_the_builtin():
    pytest.importorskip("talib")
    c = _close()
    sma = alphax.indicators.from_talib("SMA", params=["timeperiod"]).run(c, timeperiod=[5, 10])
    np.testing.assert_allclose(sma.out, alphax.indicators.ma(c, window=[5, 10]).ma, rtol=1e-12)


# --- factor engine: cross-sectional operators, WQ101, batches, streams -------


def _ohlcv(n=80, cols=5, seed=1):
    c = _close(n, cols, seed)
    rng = np.random.default_rng(seed)
    return {"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
            "volume": rng.uniform(1e5, 2e5, c.shape), "vwap": c * 1.0005}


def test_cross_sectional_factors_and_wq101():
    f = _ohlcv()
    out = alphax.FactorSet.compile({"r": "CSRank($close)", "z": "CSZScore(Mean($close, 5))"}).evaluate(f)
    ranks = np.argsort(np.argsort(f["close"], axis=1), axis=1)
    np.testing.assert_allclose(out["r"], (ranks + 1) / 5)
    np.testing.assert_allclose(np.nanmean(out["z"][10:], axis=1), 0, atol=1e-12)
    w = alphax.wq101()
    assert len(w.names) == 82 and len(alphax.wq101(all=True).names) == 101
    assert alphax.wq101_formula(101) == "((close - open) / ((high - low) + .001))"
    v = w.evaluate(f)
    np.testing.assert_allclose(v["alpha101"], (f["close"] - f["open"]) / (f["high"] - f["low"] + 0.001))


def test_factor_batches_match_and_write_parquet(tmp_path):
    f = _ohlcv(cols=7)
    s = alphax.alpha158()
    whole = s.evaluate(f)
    seen = []

    def sink(start, stop, values):
        seen.append((start, stop))
        np.testing.assert_array_equal(values["MA5"], whole["MA5"][:, start:stop])

    s.evaluate_batches(f, 3, sink)
    assert seen == [(0, 3), (3, 6), (6, 7)]
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "f.parquet"
    s.to_parquet(str(path), f, batch=3, symbols=list("ABCDEFG"))
    t = pq.read_table(path)
    assert t.num_rows == 80 * 7 and pq.ParquetFile(path).num_row_groups == 3
    np.testing.assert_array_equal(t.column("MA5").to_numpy()[80:160], whole["MA5"][:, 1])


def test_indicator_streams_match_the_batch():
    c = _close(200, 3)
    h, lo = c * 1.01, c * 0.99
    for kind, batch, kw in [
        ("rsi", alphax.indicators.rsi(c, window=14), {}),
        ("macd", alphax.indicators.macd(c), {}),
        ("atr", alphax.indicators.atr(h, lo, c), {"high": h, "low": lo}),
        ("bbands", alphax.indicators.bbands(c), {}),
    ]:
        s = alphax.indicators.Stream(kind, n_cols=3)
        rows = [s.update(c[i], **{k: v[i] for k, v in kw.items()}) for i in range(len(c))]
        for name in s.names:
            got = np.array([r[name] for r in rows])
            np.testing.assert_array_equal(got, batch.outputs[name], err_msg=f"{kind}.{name}")
    with pytest.raises(TypeError, match="no parameter"):
        alphax.indicators.Stream("rsi", span=3)
    # The binding's own defaults are the Python functions'.
    raw = alphax._alphax._IndicatorStream("bbands", 3, {})
    bb = alphax.indicators.bbands(c)
    rows = [raw.update(list(c[i]), None, None, None) for i in range(len(c))]
    np.testing.assert_array_equal(np.array([r["middle"] for r in rows]), bb.outputs["middle"])
