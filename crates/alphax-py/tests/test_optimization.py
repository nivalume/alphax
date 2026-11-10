"""Parameter search, date splits, significance tests and portfolio weights."""

import numpy as np
import pytest

import alphax
from alphax import allocation as al
from alphax import params as P
from alphax import validation as V


def _close(n=300, k=4, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.array([0.005, 0.01, 0.02, 0.03])[:k]
    return 100 * np.exp(np.cumsum(rng.normal(0.0003, vol, (n, k)), axis=0))


# --- parameters ---------------------------------------------------------------


def test_grids_random_search_and_results():
    g = P.grid(fast=[5, 10, 20], slow=[10, 20, 50], where=lambda p: p["fast"] < p["slow"])
    assert len(g) == 6 and g[0] == {"fast": 5, "slow": 10}
    r1 = P.random(5, seed=3, a=P.IntUniform(1, 3), b=P.LogUniform(0.1, 10), c=["x", "y"], d=7)
    assert r1 == P.random(5, seed=3, a=P.IntUniform(1, 3), b=P.LogUniform(0.1, 10), c=["x", "y"], d=7)
    assert all(1 <= p["a"] <= 3 and 0.1 <= p["b"] < 10 and p["c"] in "xy" and p["d"] == 7 for p in r1)
    with pytest.raises(ValueError, match="rejected"):
        P.random(1, a=[1], where=lambda p: False)

    res = P.run(lambda fast, slow, k=1: {"score": k * (slow - fast), "n": fast}, g, k=2)
    assert res.best("score") == {"fast": 5, "slow": 50}
    assert res.best("score", minimize=True) == {"fast": 5, "slow": 10}
    t = res.table()
    assert t["fast"] == [p["fast"] for p in g] and t["score"][0] == 10
    assert len(res.select(fast=5)) == 3
    threaded = P.run(lambda fast, slow: fast * slow, g, n_jobs=3)
    assert list(threaded.column()) == [p["fast"] * p["slow"] for p in g]


def test_neighbors_find_isolated_peaks():
    g = P.grid(a=[1, 2, 3], b=[1, 2, 3])
    # A spike at (2, 2) on a flat field, and a plateau corner.
    res = P.run(lambda a, b: 10.0 if (a, b) == (2, 2) else 1.0, g)
    nb = P.neighbors(res)
    i = g.index({"a": 2, "b": 2})
    assert nb["count"][i] == 4 and nb["mean"][i] == 1.0 and nb["ratio"][i] == pytest.approx(0.1)
    corner = g.index({"a": 1, "b": 1})
    assert nb["count"][corner] == 2 and nb["min"][corner] == 1.0
    mixed = P.ParamResults(g + [{"a": 2, "b": 2, "c": 5}], res.values + [3.0])
    assert P.neighbors(mixed)["count"][-1] == 0


def test_backtests_run_over_a_grid():
    c = _close()

    def sharpe(window):
        ma = np.vstack([np.full((window - 1, c.shape[1]), np.nan),
                        np.lib.stride_tricks.sliding_window_view(c, window, axis=0).mean(axis=2)])
        pf = alphax.from_signals(c, c > ma, c < ma, size=1.0)
        return float(np.nanmean(pf.sharpe_ratio()))

    res = P.run(sharpe, P.grid(window=[5, 10, 20, 40]), n_jobs=2)
    assert np.isfinite(res.column()).all()


# --- splits from dates ----------------------------------------------------------


def test_calendar_and_date_splits():
    days = np.arange("2023-01-01", "2024-12-31", dtype="datetime64[D]")
    s = V.calendar(days, "M", train=6, test=1)
    assert len(s) == 24 - 6
    first = s[0]
    assert str(days[first.train[0]]) == "2023-01-01" and str(days[first.train[-1]]) == "2023-06-30"
    assert str(days[first.test[0]]) == "2023-07-01" and str(days[first.test[-1]]) == "2023-07-31"
    grow = V.calendar(days, "Q", train=2, test=1, expanding=True)
    assert all(sp.train[0] == 0 for sp in grow) and len(grow) == 6
    r = V.from_ranges(days, [("2023-01-01", "2023-07-01", "2023-07-01", "2023-08-01")])
    np.testing.assert_array_equal(r[0].train, first.train)
    np.testing.assert_array_equal(r[0].test, first.test)
    m = V.from_masks(np.arange(10) < 5, (np.arange(10) >= 5) & (np.arange(10) < 7))
    np.testing.assert_array_equal(m[0].test, [5, 6])
    with pytest.raises(ValueError, match="both"):
        V.from_masks([True, False], [True, True])
    with pytest.raises(ValueError, match="no bars"):
        V.from_ranges(days, [("2030-01-01", "2030-02-01", "2023-01-01", "2023-02-01")])


# --- significance tests -----------------------------------------------------------


def test_permutation_test_separates_skill_from_luck():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, 500)
    skilled = np.sign(ret) * (rng.random(500) < 0.6)   # knows the sign 60% of bars
    lucky = np.sign(rng.normal(size=500))
    assert V.permutation_test(skilled, ret, n=300)["p_value"] < 0.01
    assert V.permutation_test(lucky, ret, n=300)["p_value"] > 0.05
    out = V.permutation_test(skilled, ret, statistic="sharpe", n=50, block=5, seed=1)
    assert out["null"].shape == (50,)
    again = V.permutation_test(skilled, ret, statistic="sharpe", n=50, block=5, seed=1)
    np.testing.assert_array_equal(out["null"], again["null"])


def test_signal_delay_test_runs_each_delay():
    c = _close(200, 2)

    @alphax.strategy
    def cross(data, window=10):
        close = data["close"]
        ma = np.vstack([np.full((window - 1, close.shape[1]), np.nan),
                        np.lib.stride_tricks.sliding_window_view(close, window, axis=0).mean(axis=2)])
        return close > ma, close < ma

    out = V.signal_delay_test(cross, {"close": c}, {"window": 10}, delays=(0, 1, 2), size=1.0)
    assert set(out) == {0, 1, 2} and all(v.shape == (2,) for v in out.values())
    assert not np.array_equal(out[0], out[1])


# --- portfolio weights -------------------------------------------------------------


def test_weight_functions():
    c = _close(400)
    r = c[1:] / c[:-1] - 1
    np.testing.assert_allclose(al.equal_weight(r), 0.25)
    iv = al.inverse_volatility(r)
    assert iv.sum() == pytest.approx(1) and (np.diff(iv) < 0).all()
    mv = al.min_variance(r)
    s = np.cov(r, rowvar=False)
    assert mv.sum() == pytest.approx(1) and (mv >= 0).all()
    assert mv @ s @ mv <= al.equal_weight(r) @ s @ al.equal_weight(r)
    rp = al.risk_parity(r)
    contrib = rp * (s @ rp)
    np.testing.assert_allclose(contrib / contrib.sum(), 0.25, atol=1e-6)
    ms = al.max_sharpe(r)
    assert ms.sum() == pytest.approx(1) or (ms == 0).all()


def test_risk_parity_with_negative_covariances():
    # Two assets that move against each other and a small independent one:
    # the equal-weight marginal risk of the hedged pair is negative, which the
    # multiplicative update turned into the square root of a negative number.
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.05, 60)
    r = np.column_stack([a, -0.9 * a + rng.normal(0, 0.005, 60), rng.normal(0, 0.01, 60)])
    s = np.cov(r, rowvar=False)
    assert (np.full(3, 1 / 3) * (s @ np.full(3, 1 / 3)) <= 0).any()
    rp = al.risk_parity(r)
    assert np.isfinite(rp).all() and (rp > 0).all() and rp.sum() == pytest.approx(1)
    contrib = rp * (s @ rp)
    np.testing.assert_allclose(contrib / contrib.sum(), 1 / 3, atol=1e-6)


def test_weights_on_a_schedule_and_their_backtest():
    c = _close(300)
    w = al.weights(c, al.equal_weight, every=20, lookback=30)
    rows = np.flatnonzero(~np.isnan(w[:, 0]))
    assert rows[0] == 40 and (np.diff(rows) == 20).all()
    pf = al.from_weights(c, w)
    o = pf.orders()
    assert set(np.unique(o["idx"])) <= set(rows)
    # After a rebalance the holdings are equal in value.
    t = rows[0]
    held = np.array([o["size"][(o["idx"] == t) & (o["col"] == k)].sum() for k in range(4)]) * c[t]
    np.testing.assert_allclose(held / held.sum(), 0.25, rtol=1e-9)
    days = np.arange("2024-01-01", 300, dtype="datetime64[D]")
    wm = al.weights(c, al.min_variance, freq="M", index=days, lookback=20)
    first = np.flatnonzero(~np.isnan(wm[:, 0]))
    assert all(str(days[i])[8:] == "01" for i in first)
    with pytest.raises(ValueError, match="returned 3 weights"):
        al.weights(c, lambda win: np.ones(3), every=50, lookback=10)
    with pytest.raises(ValueError, match="exactly one"):
        al.rebalance_mask(10)


def test_optimizer_wrappers_when_installed():
    pytest.importorskip("pypfopt")
    c = _close(300)
    w = al.weights(c, al.from_pypfopt("min_volatility"), every=60, lookback=120)
    assert np.nansum(w[~np.isnan(w[:, 0])], axis=1) == pytest.approx(1)


def test_shuffled_prices_keep_the_returns_and_lose_the_order():
    c = _close(200, 3)
    c[50, 1] = np.nan
    paths = V.shuffled_prices(c, 5, block=4, seed=2)
    assert paths.shape == (5, 200, 3) and np.isnan(paths[:, 50, 1]).all()
    lr = np.diff(np.log(c[:, 0]))
    np.testing.assert_allclose(np.sort(np.diff(np.log(paths[0, :, 0]))), np.sort(lr), atol=1e-12)
    np.testing.assert_allclose(paths[:, 0], np.broadcast_to(c[0], (5, 3)))
    # A trend follower on a trending series beats its shuffled versions.
    trend = 100 * np.exp(np.cumsum(np.r_[np.full(100, 0.01), np.full(100, -0.01)] +
                                   np.random.default_rng(0).normal(0, 0.002, 200)))

    def follow(p):
        r = np.diff(np.log(p))
        return float(np.sum(np.sign(r[:-1]) * r[1:]))

    assert V.shuffled_test(follow, trend, n=50)["p_value"] < 0.05
