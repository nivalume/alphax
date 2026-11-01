"""Evaluating a strategy beyond its Sharpe: hit ratio and return concentration,
drawdown recovery, resampled drawdowns, delayed positions, two-dimensional
permutation tests, the predictive power of a signal and its decay, what a
strategy adds to a book, and the charts of a return series."""

import math

import numpy as np
import pytest

import phasor
from phasor import allocation, predictions, robust, validation


def _returns(n=2000, seed=0, mean=0.0005, vol=0.01):
    return np.random.default_rng(seed).normal(mean, vol, n)


# --- return metrics -----------------------------------------------------------------------------


def test_hit_ratio_and_concentration_through_metric_rolling_and_stats():
    r = np.array([0.1, -0.1, 0.0, np.nan, 0.2])
    assert phasor.returns.metric(r, "hit_ratio") == pytest.approx(2 / 3)
    spiky = np.r_[np.full(19, 0.01), 0.5, -0.2]
    assert phasor.returns.metric(spiky, "return_concentration") == pytest.approx(0.51 / 0.69)
    assert phasor.returns.metric(spiky, "return_concentration", cutoff=1.0) == pytest.approx(1.0)
    two = np.column_stack([_returns(seed=1), _returns(seed=2)])
    hits = phasor.returns.metric(two, "hit_ratio")
    assert hits.shape == (2,) and np.all((hits > 0.4) & (hits < 0.6))
    rolled = phasor.returns.rolling(two, "hit_ratio", 50)
    assert rolled[49:, 0] == pytest.approx([phasor.returns.metric(two[t - 49 : t + 1, 0], "hit_ratio")
                                            for t in range(49, len(two))])
    st = phasor.returns.stats(two[:, 0])
    assert {"hit_ratio", "return_concentration", "max_drawdown_recovery"} <= set(st)


def test_the_deepest_drawdowns_recovery_in_bars():
    # Up, a fall over two bars, back to the peak three bars after the valley.
    r = np.array([0.1, -0.1, -0.1, 0.05, 0.05, 0.13, 0.01])
    st = phasor.returns.stats(r)
    d = phasor.returns.drawdowns(r)
    deepest = int(np.argmin(d["drawdown"]))
    assert d["recovered"][deepest]
    assert st["max_drawdown_recovery"] == d["recovery_duration"][deepest] == 3
    assert math.isnan(phasor.returns.stats(np.array([0.1, -0.2, 0.01]))["max_drawdown_recovery"])


# --- resampled metrics --------------------------------------------------------------------------


def test_bootstrap_puts_the_history_inside_its_distribution():
    r = _returns()
    found = robust.bootstrap(r, n=300, block=5, seed=1)
    dd = found["max_drawdown"]
    assert dd["observed"] == pytest.approx(phasor.returns.metric(r, "max_drawdown"))
    q = dd["quantiles"]
    assert q[0.01] <= q[0.05] <= q[0.5] <= q[0.95] <= 0
    assert q[0.01] < dd["observed"] < 0
    assert dd["values"].shape == (300,)
    sharpe = found["sharpe_ratio"]["quantiles"]
    assert sharpe[0.05] < found["sharpe_ratio"]["observed"] < sharpe[0.95]
    again = robust.bootstrap(r, n=300, block=5, seed=1)
    assert np.array_equal(again["max_drawdown"]["values"], dd["values"])


def test_bootstrap_of_a_too_short_series_is_nan():
    found = robust.bootstrap([0.1], n=10)
    assert math.isnan(found["max_drawdown"]["observed"])
    with pytest.raises(ValueError):
        robust.bootstrap(_returns(), n=0)


# --- delayed positions and permutation ----------------------------------------------------------


def test_position_delay_test_loses_a_one_bar_edge():
    rng = np.random.default_rng(3)
    ret = rng.normal(0, 0.01, (3000, 3))
    known = np.sign(ret)  # held over the very return it knows
    curve = validation.position_delay_test(known, ret, delays=(0, 1, 2))
    assert curve[0] > 10 and abs(curve[1]) < 2 and abs(curve[2]) < 2
    # A slow edge survives a delay.
    trend = np.cumsum(rng.normal(0, 0.001, (3000, 3)), axis=0)
    slow = ret + 0.002 * np.sign(trend)
    slow_curve = validation.position_delay_test(np.sign(trend), slow, delays=(0, 1, 2))
    assert slow_curve[2] > 0.8 * slow_curve[0] > 0


def test_position_delay_test_charges_costs_and_takes_one_column():
    ret = np.full(100, 0.001)
    flips = np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    free = validation.position_delay_test(np.ones(100), ret, delays=(0,), metric="total_return")[0]
    paid = validation.position_delay_test(flips, ret, delays=(0,), cost=0.01, metric="total_return")[0]
    assert free > 0 > paid
    with pytest.raises(ValueError):
        validation.position_delay_test(np.ones(10), np.ones(11))


def test_permutation_test_in_two_dimensions():
    rng = np.random.default_rng(4)
    ret = rng.normal(0, 0.01, (400, 3))
    known = np.sign(ret)
    p_known = validation.permutation_test(known, ret, statistic="sharpe", n=100, block=5)["p_value"]
    assert p_known < 0.02
    noise = np.sign(rng.normal(size=(400, 3)))
    assert validation.permutation_test(noise, ret, statistic="sharpe", n=100)["p_value"] > 0.02
    # One column in two dimensions is the one-dimensional test.
    one = validation.permutation_test(noise[:, 0], ret[:, 0], n=50, block=3, seed=7)
    two = validation.permutation_test(noise[:, :1], ret[:, :1], n=50, block=3, seed=7)
    assert one["observed"] == pytest.approx(two["observed"]) and np.allclose(one["null"], two["null"])
    with pytest.raises(ValueError):
        validation.permutation_test(noise, ret[:, :2])


# --- predictive power ---------------------------------------------------------------------------


def test_power_over_time_recovers_a_known_r_squared():
    rng = np.random.default_rng(5)
    signal = rng.normal(size=20000)
    rho = 0.2
    fwd = rho * signal + math.sqrt(1 - rho * rho) * rng.normal(size=20000)
    found = predictions.power(signal, fwd, quantiles=5)
    assert found["correlation"] == pytest.approx(rho, abs=0.02)
    assert found["r_squared"] == pytest.approx(rho * rho, abs=0.01)
    means = found["quantile_returns"]
    assert list(means) == [1, 2, 3, 4, 5] and found["monotonicity"] == pytest.approx(1.0)
    assert sum(found["quantile_counts"].values()) == found["observations"] == 20000


def test_tied_predictions_share_a_group():
    pos = np.repeat([-1.0, 0.0, 1.0], 100)
    fwd = np.repeat([-0.01, 0.0, 0.01], 100)
    found = predictions.power(pos, fwd, quantiles=5)
    assert len(found["quantile_returns"]) == 3
    assert sorted(found["quantile_counts"].values()) == [100, 100, 100]
    assert sorted(found["quantile_predictions"].values()) == [-1.0, 0.0, 1.0]
    assert found["monotonicity"] == pytest.approx(1.0)


def test_power_across_instruments_is_the_information_coefficient():
    rng = np.random.default_rng(6)
    signal = rng.normal(size=(500, 40))
    fwd = 0.1 * signal + rng.normal(size=(500, 40))
    signal[::7, :39] = np.nan  # bars with one instrument are skipped
    found = predictions.power(signal, fwd, axis="cross_section")
    assert found["correlation"] == pytest.approx(0.1, abs=0.02)
    assert found["periods"] == 500 - len(range(0, 500, 7))
    assert found["information_ratio"] > 0.5 and found["monotonicity"] > 0.8
    with pytest.raises(ValueError):
        predictions.power(signal[:, 0], fwd[:, 0], axis="cross_section")


def test_decay_follows_the_direction_of_the_prediction():
    close = np.r_[np.linspace(100, 200, 50), np.linspace(200, 100, 50)][:, None]
    up = np.ones_like(close)
    found = predictions.decay(up, close, horizons=(1, 10))
    assert set(found) == {1, 10}
    assert math.isnan(predictions.decay(np.zeros_like(close), close, horizons=(1,))[1])  # nothing predicted
    rising = predictions.decay(np.r_[np.ones(50), np.zeros(50)][:, None], close, horizons=(1, 5))
    assert rising[1] > 0 and rising[5] > rising[1]


# --- what a strategy adds -----------------------------------------------------------------------


def test_marginal_contribution_rejects_a_copy_and_values_a_diversifier():
    rng = np.random.default_rng(8)
    a = rng.normal(0.001, 0.01, 3000)
    b = rng.normal(0.001, 0.01, 3000)
    copy = allocation.marginal_contribution(a[:, None], 2 * a)
    assert copy["max_abs_correlation"] == pytest.approx(1.0)
    assert copy["gain"] == pytest.approx(0.0, abs=1e-9)
    fresh = allocation.marginal_contribution(a[:, None], b)
    assert abs(fresh["max_abs_correlation"]) < 0.1
    assert fresh["combined_sharpe"] > fresh["book_sharpe"] and fresh["gain"] > 0.5
    first = allocation.marginal_contribution(None, b)
    assert first["gain"] == first["candidate_sharpe"] and first["correlations"] == []
    rows = np.arange(3000) >= 1500
    part = allocation.marginal_contribution(np.column_stack([a, b]), a + b, fit=~rows, rows=rows)
    assert len(part["correlations"]) == 2 and part["correlation_to_book"] > 0.95


# --- charts -------------------------------------------------------------------------------------


def test_charts_of_a_return_series_and_of_bars():
    pytest.importorskip("plotly")
    r = np.column_stack([_returns(500), _returns(500, seed=1)])
    fig = phasor.plotting.returns(r, names=["a", "b"], segments={"train": (0, 300), "test": (300, None)})
    assert len(fig.data) == 4 and len(fig.layout.shapes) >= 2
    fig = phasor.plotting.bars({0: 1.2, 1: -0.3}, title="Sharpe by delay")
    assert list(fig.data[0].x) == ["0", "1"]
