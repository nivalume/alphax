"""Python shell of factor evaluation: layout, shapes, and the report.

The engine tests the numbers against the golden fixture. These tests check
the shell: a 1-D input comes back 1-D, a C-ordered panel is read as the
F-ordered one, the report's axes are (factor, horizon, bar), and one thread
matches several.
"""

import numpy as np
import pytest

import phasor
from phasor import factor_eval


def _cell(table, factor, horizon, column):
    if hasattr(table, "loc"):
        return table.loc[(factor, horizon), column]
    return table[(factor, horizon)][column]


def _as_dict(table):
    return table.to_dict() if hasattr(table, "to_dict") else table


def _same(left, right):
    """Bit-identical numbers, with NaN matching NaN. Walks the summary tables."""
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
        return
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        if np.isnan(left) and np.isnan(right):
            return
        assert left == right
        return
    assert left == right


def test_forward_returns_shift_with_lag_in_either_layout():
    close = np.array([1.0, 2.0, 4.0, 8.0])
    out = factor_eval.forward_returns(close, horizons=(1, 2), lag=1)
    assert out[1].shape == (4,)
    np.testing.assert_allclose(out[1][:2], [1.0, 1.0])
    assert np.isnan(out[1][2:]).all()
    np.testing.assert_allclose(out[2][0], 3.0)
    assert np.isnan(out[2][1:]).all()

    same_bar = factor_eval.forward_returns(close, horizons=(1,), lag=0)[1]
    np.testing.assert_allclose(same_bar[:3], [1.0, 1.0, 1.0])
    assert np.isnan(same_bar[3])

    panel = np.column_stack([close, close * 10.0])
    both = factor_eval.forward_returns(panel, horizons=(1,), lag=0)[1]
    assert both.shape == (4, 2)
    np.testing.assert_allclose(both[:3], 1.0)
    assert np.isnan(both[3]).all()
    np.testing.assert_array_equal(both[:, 0], same_bar)

    contiguous = np.ascontiguousarray(panel)
    assert not contiguous.flags.f_contiguous
    np.testing.assert_array_equal(factor_eval.forward_returns(contiguous, horizons=(1,), lag=0)[1], both)
    with pytest.raises(ValueError, match="C-ordered"):
        phasor._phasor.factor_forward_returns(contiguous, 1, 0)


def test_cs_ic_and_report_score_a_factor_against_its_own_return():
    rows, cols = 50, 8
    bar = np.arange(rows)[:, None]
    symbol = np.arange(cols)[None, :]
    close = 100.0 * np.exp(0.01 * (symbol + 1) * bar)
    horizons = (1, 5, 10)
    forwards = factor_eval.forward_returns(close, horizons=horizons, lag=1)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=forwards[1].shape)

    report = factor_eval.evaluate(
        {"good": forwards[1], "noise": noise},
        close,
        horizons=horizons,
        lag=1,
        min_names=4,
        threads=1,
        index=np.arange(rows),
    )
    assert "factor_eval" in phasor.__all__
    assert report.names == ["good", "noise"]
    assert report.pearson.shape == (2, 3, rows)
    assert report.spearman.shape == (2, 3, rows)
    np.testing.assert_allclose(report.ic("good", 1, "pearson"), report.pearson[0, 0])
    finite = np.isfinite(report.ic("good", 1))
    assert finite.sum() == rows - (1 + 1)
    np.testing.assert_allclose(report.ic("good", 1, "rank")[finite], 1.0)
    rank_5 = report.ic("good", 5, "rank")
    np.testing.assert_allclose(rank_5[np.isfinite(rank_5)], 1.0)

    pearson, spearman = factor_eval.cs_ic([forwards[1], noise], [forwards[1], forwards[5]], min_names=4, threads=1)
    assert pearson.shape == (2, 2, rows)
    np.testing.assert_array_equal(pearson[0, 0], report.pearson[0, 0])
    np.testing.assert_array_equal(spearman[0, 1], report.spearman[0, 1])

    table = report.summary()
    columns = list(table.columns) if hasattr(table, "columns") else list(next(iter(table.values())))
    assert columns == ["ic", "rank_ic", "icir", "t", "t_nw", "p_nw", "hit", "n"]
    np.testing.assert_allclose(_cell(table, "good", 1, "ic"), 1.0)
    np.testing.assert_allclose(_cell(table, "good", 1, "rank_ic"), 1.0)
    assert _cell(table, "good", 1, "ic") > _cell(table, "noise", 1, "ic")
    np.testing.assert_allclose(_cell(table, "good", 5, "rank_ic"), 1.0)
    assert _cell(table, "good", 1, "n") == int(finite.sum())
    assert _cell(table, "good", 1, "hit") == 1.0

    decay = _as_dict(report.decay("good"))
    assert list(decay) == [1, 5, 10]
    np.testing.assert_allclose([decay[1], decay[5], decay[10]], 1.0)
    np.testing.assert_allclose(_as_dict(report.decay("good", method="pearson"))[1], 1.0)

    again = factor_eval.evaluate(
        {"good": forwards[1], "noise": noise},
        forward=forwards,
        horizons=horizons,
        min_names=4,
    )
    np.testing.assert_array_equal(again.ic("good", 1, "pearson"), report.ic("good", 1, "pearson"))

    hand = factor_eval.summarize([0.1, -0.05, 0.2, 0.03, 0.07], lags=0)
    assert hand["n"] == 5 and hand["hit"] == 0.8
    assert phasor._phasor.factor_auto_lags(100, 1) == 4
    assert phasor._phasor.factor_auto_lags(100, 20) == 19
    assert factor_eval.summarize([0.1, np.nan, 0.2], horizon=1)["n"] == 2

    with pytest.raises(ValueError, match="close or forward"):
        factor_eval.evaluate(forwards[1])
    with pytest.raises(ValueError, match="only one"):
        factor_eval.evaluate({"good": forwards[1]}, close, forward=forwards)
    with pytest.raises(ValueError, match="no factor"):
        report.ic("missing")


def test_ts_ic_and_rolling_ic_follow_each_symbol():
    x = np.arange(30.0)
    y = 3.0 * x + 1.0
    ic = factor_eval.ts_ic(x, y, method="pearson")
    assert ic.shape == (1,)
    np.testing.assert_allclose(ic, 1.0)
    np.testing.assert_allclose(factor_eval.ts_ic(x, -y, method="rank"), -1.0)

    panel_x = np.column_stack([x, x[::-1]])
    panel_y = np.column_stack([y, x])
    both = factor_eval.ts_ic(panel_x, panel_y, method="spearman", threads=1)
    assert both.shape == (2,)
    np.testing.assert_allclose(both, [1.0, -1.0])
    np.testing.assert_array_equal(both, factor_eval.ts_ic(panel_x, panel_y, method="spearman", threads=4))

    rolling = factor_eval.ts_rolling_ic(x, y, window=5, min_periods=5, method="pearson")
    assert rolling.shape == (30,)
    assert np.isnan(rolling[:4]).all()
    np.testing.assert_allclose(rolling[4:], 1.0)
    panel = factor_eval.ts_rolling_ic(panel_x, panel_y, window=8, min_periods=4, method="rank", threads=1)
    assert panel.shape == (30, 2)
    np.testing.assert_array_equal(
        panel,
        factor_eval.ts_rolling_ic(panel_x, panel_y, window=8, min_periods=4, method="rank", threads=4),
    )
    with pytest.raises(ValueError, match="method"):
        factor_eval.ts_ic(x, y, method="kendall")


def test_block_ic_groups_rows_that_are_not_adjacent():
    # The engine fixture: block 7 is x = 1, 2, 3 against y = 1, 2, 3;
    # block 3 is x = 5, 4, 9 against y = 0, 1, -4.
    x = np.array([1.0, 5.0, 2.0, 4.0, 3.0, 9.0])
    y = np.array([1.0, 0.0, 2.0, 1.0, 3.0, -4.0])
    blocks = np.array([7, 3, 7, 3, 7, 3])
    ids, ic = factor_eval.ts_block_ic(x, y, blocks, min_periods=2, method="pearson")
    assert list(ids) == [3, 7]
    assert ic.shape == (2,)
    np.testing.assert_allclose(ic, [-1.0, 1.0])

    rows, cols = 40, 3
    panel = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols, order="F")
    months = np.arange(rows) // 10
    ids, ic = factor_eval.ts_block_ic(panel, panel * 2.0 + 1.0, months, min_periods=2, threads=4)
    assert list(ids) == [0, 1, 2, 3]
    assert ic.shape == (4, cols)
    np.testing.assert_allclose(ic, 1.0)
    with pytest.raises(ValueError, match="one per row"):
        factor_eval.ts_block_ic(x, y, blocks[:-1])


def test_thread_invariance_of_evaluate():
    rng = np.random.default_rng(11)
    rows, cols = 50, 8
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(rows, cols)), axis=0))
    good = factor_eval.forward_returns(close, horizons=(1,), lag=1)[1]
    noise = rng.normal(size=good.shape)
    factors = {"good": good, "noise": noise}

    serial = factor_eval.evaluate(factors, close, horizons=(1, 5, 10), min_names=4, threads=1)
    parallel = factor_eval.evaluate(factors, close, horizons=(1, 5, 10), min_names=4, threads=4)
    for name in factors:
        for horizon in (1, 5, 10):
            for method in ("pearson", "rank"):
                np.testing.assert_array_equal(serial.ic(name, horizon, method), parallel.ic(name, horizon, method))
    _same(_as_dict(serial.summary()), _as_dict(parallel.summary()))
    assert _cell(serial.summary(), "good", 1, "ic") > 0.0


def _values(table):
    return np.asarray(table.to_numpy() if hasattr(table, "to_numpy") else table)


def test_quantile_labels_handle_ties_nans_and_short_rows():
    factor = np.array(
        [
            [1.0, 3.0, 2.0, np.nan],
            [1.0, 1.0, 2.0, 3.0],
            [5.0, np.nan, 6.0, np.nan],
        ]
    )
    labels = factor_eval.quantile_labels(factor, q=2)
    assert labels.dtype == np.int16
    assert labels.shape == (3, 4)
    np.testing.assert_array_equal(
        labels,
        [
            [0, 1, 1, -1],
            [0, 0, 1, 1],
            [0, -1, 1, -1],
        ],
    )
    wide = factor_eval.quantile_labels(factor, q=4)
    np.testing.assert_array_equal(wide[1], [1, 1, 2, 3])
    np.testing.assert_array_equal(wide[2], [-1, -1, -1, -1])

    contiguous = np.ascontiguousarray(factor)
    assert not contiguous.flags.f_contiguous
    np.testing.assert_array_equal(factor_eval.quantile_labels(contiguous, q=2), labels)
    with pytest.raises(ValueError, match="C-ordered"):
        phasor._phasor.factor_quantile_labels(contiguous, 2)
    with pytest.raises(ValueError, match="1 or more"):
        factor_eval.quantile_labels(factor, q=0)


def test_quantile_returns_long_short_and_monotonicity():
    # Square on purpose: a C-ordered reading would swap the two columns.
    labels = np.array([[0, 1], [1, 0]], dtype=np.int16)
    forward = np.array([[0.1, 0.4], [0.2, 0.3]])
    out = factor_eval.quantile_returns(labels, forward, q=2)
    np.testing.assert_allclose(out, [[0.1, 0.4], [0.3, 0.2]])
    np.testing.assert_allclose(factor_eval.long_short_returns(out, q=2), [0.3, -0.1])
    same = factor_eval.quantile_returns(np.ascontiguousarray(labels), np.ascontiguousarray(forward), q=2)
    np.testing.assert_allclose(same, out)
    with pytest.raises(ValueError, match="C-ordered"):
        phasor._phasor.factor_quantile_returns(
            np.ascontiguousarray(labels), np.asfortranarray(forward), 2, None
        )

    grouped = np.array([[0, 0, 1, 1]], dtype=np.int16)
    returns = np.array([[0.1, 0.2, 0.3, 0.4]])
    weights = np.array([[1.0, 3.0, 1.0, 1.0]])
    weighted = factor_eval.quantile_returns(grouped, returns, q=2, weights=weights)
    np.testing.assert_allclose(weighted, [[0.175, 0.35]])
    flat = factor_eval.quantile_returns(grouped, returns, q=2, weights=np.array([[1.0, 1.0, 0.0, 0.0]]))
    assert np.isnan(flat[0, 1])
    np.testing.assert_allclose(flat[0, 0], 0.15)

    assert factor_eval.monotonicity([0.1, 0.3, 0.2]) == pytest.approx(0.5)
    assert factor_eval.monotonicity([0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert np.isnan(factor_eval.monotonicity([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="groups"):
        factor_eval.long_short_returns(out, q=3)
    with pytest.raises(ValueError, match="shape"):
        factor_eval.quantile_returns(labels[:, :1], forward, q=1)


def test_turnover_and_rank_autocorr():
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int16,
    )
    turnover = factor_eval.quantile_turnover(labels, q=2, lag=1)
    assert turnover.shape == (3, 2)
    assert np.isnan(turnover[0]).all()
    np.testing.assert_allclose(turnover[1], [0.0, 1.0 / 3.0])
    np.testing.assert_allclose(turnover[2], [1.0, 0.5])

    factor = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, np.nan, 3.0],
            [5.0, 5.0, 5.0],
        ]
    )
    autocorr = factor_eval.rank_autocorr(factor, lag=1)
    assert autocorr.shape == (4,)
    assert np.isnan(autocorr[0]) and np.isnan(autocorr[3])
    np.testing.assert_allclose(autocorr[1:3], [-1.0, -1.0])
    np.testing.assert_allclose(factor_eval.rank_autocorr(factor, lag=2)[2], 1.0)


def test_report_quantiles_and_backtest():
    rows, cols = 40, 8
    symbol = np.arange(cols, dtype=float)
    bar = np.arange(rows)[:, None]
    close = 100.0 * np.exp(0.01 * (symbol + 1) * bar)
    factor = np.broadcast_to(symbol, (rows, cols)).copy()
    report = factor_eval.evaluate(
        {"rank": factor},
        close,
        horizons=(1, 5),
        min_names=4,
        index=np.arange(rows),
    )
    table = report.quantile_returns("rank", h=1, q=4)
    values = _values(table)
    assert values.shape == (rows, 4)
    if hasattr(table, "columns"):
        assert list(table.columns) == [0, 1, 2, 3]
        assert len(table.index) == rows
    finite = np.isfinite(values[:, 0])
    assert finite.sum() == rows - 2
    means = np.array([values[finite, g].mean() for g in range(4)])
    assert np.all(np.diff(means) > 0)
    np.testing.assert_allclose(report.long_short("rank", h=1, q=4), values[:, -1] - values[:, 0])
    assert report.monotonicity("rank", h=1, q=4) == pytest.approx(1.0)

    turnover = _values(report.turnover("rank", q=4, lag=1))
    assert turnover.shape == (rows, 4)
    assert np.isnan(turnover[0]).all()
    np.testing.assert_allclose(turnover[1:], 0.0)
    autocorr = report.rank_autocorr("rank", lag=1)
    assert np.isnan(autocorr[0])
    np.testing.assert_allclose(autocorr[1:], 1.0)

    with pytest.raises(ValueError, match="no factor"):
        report.quantile_returns("missing")
    bare = factor_eval.FactorReport(["rank"], [1], np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), {})
    with pytest.raises(ValueError, match="no values"):
        bare.quantile_returns("rank")

    result = factor_eval.quantile_backtest(
        close,
        factor,
        quantiles=4,
        rebalance=5,
        lag=1,
        long_short=False,
        init_cash=1_000_000.0,
        threads=1,
    )
    assert result.n_groups == 4
    assert result.portfolio.cash_sharing
    assert result.portfolio.n_groups == 4
    equity = result.values()
    assert equity.shape == (rows, 4)
    np.testing.assert_allclose(equity[0], 1_000_000.0)
    assert np.isfinite(equity).all()
    stats = result.stats()
    if hasattr(stats, "loc"):
        assert list(stats.columns) == ["total_return", "sharpe_ratio", "max_drawdown"]
        assert stats.loc[3, "total_return"] > stats.loc[0, "total_return"]
    else:
        assert stats[3]["total_return"] > stats[0]["total_return"]

    both = factor_eval.quantile_backtest(
        close,
        factor,
        quantiles=4,
        rebalance=5,
        lag=1,
        long_short=True,
        fees=0.001,
        slippage=0.0,
    )
    assert both.n_groups == 5
    assert both.values().shape == (rows, 5)
    assert np.isfinite(both.values()).all()
    both_stats = both.stats()
    if hasattr(both_stats, "columns"):
        assert list(both_stats.columns) == ["total_return", "sharpe_ratio", "max_drawdown"]
        assert len(both_stats) == 5
    else:
        assert set(both_stats[0]) == {"total_return", "sharpe_ratio", "max_drawdown"}


def test_factor_plots_are_plotly_figures():
    pytest.importorskip("plotly")
    rows, cols = 30, 6
    symbol = np.arange(cols, dtype=float)
    bar = np.arange(rows)[:, None]
    close = 100.0 * np.exp(0.01 * (symbol + 1) * bar)
    factor = np.broadcast_to(symbol, (rows, cols)).copy()
    report = factor_eval.evaluate({"rank": factor}, close, horizons=(1, 5), min_names=3, index=np.arange(rows))

    ic = phasor.plotting.factor_ic(report, "rank", h=1, window=5)
    assert len(ic.data) == 2
    np.testing.assert_allclose(ic.data[0].y, report.ic("rank", 1), equal_nan=True)
    assert np.isnan(np.asarray(ic.data[1].y, dtype=float)[:4]).all()

    decay = phasor.plotting.factor_decay(report, "rank")
    assert len(decay.data) >= 1
    assert list(decay.data[0].x) == [1, 5]

    quantiles = phasor.plotting.factor_quantiles(report, "rank", h=1, q=3)
    assert len(quantiles.data) == 4  # one bar chart plus one line per group

    turnover = phasor.plotting.factor_turnover(report, "rank", q=3, lag=1)
    assert len(turnover.data) == 3
    assert "factor_ic" in phasor.plotting.__all__
    assert "factor_decay" in phasor.plotting.__all__
    assert "factor_quantiles" in phasor.plotting.__all__
    assert "factor_turnover" in phasor.plotting.__all__


def _entry(table, row, column):
    """One cell of a DataFrame or of a dict of dicts."""
    if hasattr(table, "loc"):
        return table.loc[row, column]
    return table[row][column]


def _float_matrix(value):
    """A correlation matrix as a float64 array, with or without pandas."""
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=np.float64)
    names = list(value)
    return np.array([[value[a][b] for b in names] for a in names], dtype=np.float64)


def test_neutralize_removes_a_constant_and_is_orthogonal():
    y = np.array([[1.0, 2.0, 3.0, 6.0], [10.0, 0.0, 20.0, 10.0]])
    expected = y - y.mean(axis=1, keepdims=True)
    for exposure in (np.ones_like(y), np.zeros_like(y), np.full_like(y, 0.1)):
        out = factor_eval.neutralize(y, exposures=[exposure], threads=1)
        np.testing.assert_allclose(out, expected, atol=1e-10)
    contiguous = np.ascontiguousarray(y)
    assert not contiguous.flags.f_contiguous
    np.testing.assert_allclose(
        factor_eval.neutralize(contiguous, exposures={"level": np.ones_like(y)}),
        expected,
        atol=1e-10,
    )
    series = np.array([1.0, 2.0, 4.0])
    plain = factor_eval.neutralize(series)
    assert plain.shape == (3,)
    np.testing.assert_array_equal(plain, series)

    rng = np.random.default_rng(0)
    rows, cols = 25, 36
    raw = rng.normal(size=(rows, cols))
    x1 = rng.normal(size=(rows, cols))
    x2 = rng.normal(size=(rows, cols))
    groups = np.tile((np.arange(cols) % 3).astype(np.float64), (rows, 1))
    out = factor_eval.neutralize(raw, groups=groups, exposures=[x1, x2], threads=1)
    again = factor_eval.neutralize(raw, groups=groups, exposures=[x1, x2], threads=4)
    np.testing.assert_array_equal(np.ascontiguousarray(out).view(np.uint64), np.ascontiguousarray(again).view(np.uint64))
    for label in (0.0, 1.0, 2.0):
        np.testing.assert_allclose(out[:, groups[0] == label].mean(axis=1), 0.0, atol=1e-8)
    for exposure in (x1, x2):
        for t in range(rows):
            corr = np.corrcoef(out[t], exposure[t])[0, 1]
            assert abs(corr) < 1e-8


def test_fama_macbeth_recovers_known_slopes():
    rng = np.random.default_rng(3)
    rows, cols = 40, 80
    f1 = rng.normal(size=(rows, cols))
    f2 = rng.normal(size=(rows, cols))
    fwd = 2.0 * f1 - 1.5 * f2 + rng.normal(scale=0.05, size=(rows, cols))
    report = factor_eval.fama_macbeth(fwd, {"f1": f1, "f2": f2}, threads=1)
    other = factor_eval.fama_macbeth(fwd, {"f1": f1, "f2": f2}, threads=4)
    left = np.column_stack([np.asarray(report.coef[name], dtype=np.float64) for name in report.names])
    right = np.column_stack([np.asarray(other.coef[name], dtype=np.float64) for name in other.names])
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
    assert report.names == ["intercept", "f1", "f2"]
    np.testing.assert_allclose(np.mean(left[:, 1]), 2.0, atol=0.05)
    np.testing.assert_allclose(np.mean(left[:, 2]), -1.5, atol=0.05)
    summary = report.summary()
    assert _entry(summary, "f1", "mean") == pytest.approx(2.0, abs=0.05)
    assert _entry(summary, "f2", "mean") == pytest.approx(-1.5, abs=0.05)
    assert _entry(summary, "f1", "t_nw") > 2.0
    assert np.asarray(report.r2).shape == (rows,)


def test_correlation_and_max_correlation_find_copies():
    rng = np.random.default_rng(1)
    rows, cols = 20, 15
    a = rng.normal(size=(rows, cols))
    b = rng.normal(size=(rows, cols))
    neg = -a
    matrix = factor_eval.correlation({"a": a, "b": b, "neg": neg}, threads=1)
    again = factor_eval.correlation({"a": a, "b": b, "neg": neg}, threads=3)
    np.testing.assert_array_equal(
        np.ascontiguousarray(_float_matrix(matrix)).view(np.uint64),
        np.ascontiguousarray(_float_matrix(again)).view(np.uint64),
    )
    assert _entry(matrix, "a", "a") == pytest.approx(1.0)
    assert _entry(matrix, "b", "b") == pytest.approx(1.0)
    assert _entry(matrix, "a", "neg") == pytest.approx(-1.0)
    assert _entry(matrix, "neg", "a") == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="mode"):
        factor_eval.correlation({"a": a}, mode="ts")

    found = factor_eval.max_correlation({"same": a, "flip": neg, "other": b}, {"A": a, "B": b}, threads=1)
    assert _entry(found, "same", "closest") == "A"
    assert _entry(found, "same", "max_abs_corr") == pytest.approx(1.0)
    assert _entry(found, "flip", "closest") == "A"
    assert _entry(found, "flip", "max_abs_corr") == pytest.approx(1.0)
    assert _entry(found, "other", "closest") == "B"
    assert _entry(found, "other", "max_abs_corr") == pytest.approx(1.0)


def test_evaluate_adds_neutral_rank_ic():
    rng = np.random.default_rng(7)
    rows, cols = 50, 30
    exposure = rng.normal(size=(rows, cols))
    alpha = rng.normal(size=(rows, cols))
    factor = alpha + 8.0 * exposure
    groups = np.tile((np.arange(cols) % 3).astype(np.float64), (rows, 1))
    report = factor_eval.evaluate(
        {"contaminated": factor, "alpha": alpha},
        forward={1: alpha},
        horizons=(1,),
        min_names=8,
        groups=groups,
        exposures={"exposure": exposure},
        threads=1,
    )
    table = report.summary()
    columns = list(table.columns) if hasattr(table, "columns") else list(next(iter(table.values())))
    assert columns[-1] == "neutral_rank_ic"
    neutral = float(_cell(table, "contaminated", 1, "neutral_rank_ic"))
    raw = float(_cell(table, "contaminated", 1, "rank_ic"))
    assert neutral > 0.5
    assert neutral > raw
    assert float(_cell(table, "alpha", 1, "neutral_rank_ic")) > 0.5


def test_placebo_matches_across_threads_and_rejects_a_real_factor():
    rng = np.random.default_rng(0)
    rows, cols = 40, 12
    forward = rng.normal(size=(rows, cols))
    factor = forward + rng.normal(scale=0.05, size=(rows, cols))
    real = factor_eval.placebo(factor, forward, draws=30, seed=4, mode="cs", min_names=4, threads=1)
    again = factor_eval.placebo(factor, forward, draws=30, seed=4, mode="cs", min_names=4, threads=3)
    assert real["null_distribution"].shape == (30,)
    np.testing.assert_array_equal(real["null_distribution"].view(np.uint64), again["null_distribution"].view(np.uint64))
    assert real["observed"] == again["observed"]
    assert real["p_value"] == again["p_value"]
    assert real["observed"] > 0.5
    assert real["p_value"] < 0.05
    assert "placebo" in factor_eval.__all__
    assert "adjust_pvalues" in factor_eval.__all__

    shifted = factor_eval.placebo(factor, forward, draws=8, seed=4, mode="ts", min_periods=5, threads=1)
    shifted_again = factor_eval.placebo(factor, forward, draws=8, seed=4, mode="ts", min_periods=5, threads=4)
    np.testing.assert_array_equal(
        shifted["null_distribution"].view(np.uint64),
        shifted_again["null_distribution"].view(np.uint64),
    )
    assert shifted["observed"] == shifted_again["observed"]
    assert shifted["null_distribution"].shape == (8,)

    empty = factor_eval.placebo(factor, forward, draws=0, seed=0, mode="cs", min_names=4)
    assert empty["null_distribution"].shape == (0,)
    assert empty["p_value"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="mode"):
        factor_eval.placebo(factor, forward, mode="block")
    with pytest.raises(ValueError, match="seed"):
        factor_eval.placebo(factor, forward, seed=-1)


def test_adjust_pvalues_bonferroni_holm_and_bh():
    p = np.array([0.01, 0.04, 0.03])
    np.testing.assert_allclose(factor_eval.adjust_pvalues(p, method="bonferroni"), [0.03, 0.12, 0.09])
    np.testing.assert_allclose(factor_eval.adjust_pvalues([0.4, 0.5], method="bonferroni"), [0.8, 1.0])
    # Sorted 0.01, 0.03, 0.04. Holm multipliers 3, 2, 1 then a cumulative max.
    np.testing.assert_allclose(factor_eval.adjust_pvalues(p, method="holm"), [0.03, 0.06, 0.06])
    # BH: 0.03, 0.045, 0.04, then a cumulative min from the right -> 0.03, 0.04, 0.04.
    bh = factor_eval.adjust_pvalues(p, method="bh")
    np.testing.assert_allclose(bh, [0.03, 0.04, 0.04])
    np.testing.assert_allclose(factor_eval.adjust_pvalues(p, method="fdr"), bh)

    square = np.array([[0.01, 0.04], [0.03, 0.5]])
    assert factor_eval.adjust_pvalues(square, method="bh").shape == square.shape
    with_nan = factor_eval.adjust_pvalues([0.01, np.nan, 0.04], method="bonferroni")
    assert np.isnan(with_nan[1])
    np.testing.assert_allclose(with_nan[np.isfinite(with_nan)], [0.02, 0.08])
    assert factor_eval.adjust_pvalues([]).shape == (0,)
    original = p.copy()
    factor_eval.adjust_pvalues(original, method="holm")
    np.testing.assert_array_equal(original, p)
    with pytest.raises(ValueError, match="method"):
        factor_eval.adjust_pvalues(p, method="by")
