"""Initial positions, init_cash="auto", negative prices, continuous futures
and currency conversion through the Python API."""

import numpy as np
import pytest

import alphax

N = np.nan


def col(*values):
    return np.array(values, dtype=float).reshape(-1, 1)


def test_an_initial_position_is_held_from_the_first_bar():
    # 1000 cash and 10 shares at 100, sold in two steps.
    pf = alphax.from_orders(
        col(100, 105, 110), col(N, -4, -6), init_cash=1000.0, init_position=10.0, init_price=100.0
    )
    np.testing.assert_array_equal(pf.cash().ravel(), [1000, 1420, 2080])
    np.testing.assert_array_equal(pf.value().ravel(), [2000, 2050, 2080])
    o = pf.orders()
    np.testing.assert_array_equal(o["idx"], [0, 1, 2])
    np.testing.assert_array_equal(o["price"], [100, 105, 110])
    c = pf.cash_records()
    np.testing.assert_array_equal(c["amount"], [1000.0])
    # The position counts as capital: bought at 95, the first bar earns 5 a share.
    pf = alphax.from_orders(col(100, 105, 110), col(N, N, N), init_cash=1000.0,
                            init_position=10.0, init_price=95.0)
    assert pf.returns().ravel()[0] == pytest.approx(50 / 1950)
    np.testing.assert_array_equal(pf.trades()["entry_price"], [95.0])


def test_an_initial_position_in_from_signals_and_on_margin():
    # Held from the start: an exit closes it, an entry adds nothing.
    pf = alphax.from_signals(col(100, 101, 102), np.array([True, False, False]),
                             np.array([False, False, True]), init_cash=0.0, init_position=2.0)
    o = pf.orders()
    np.testing.assert_array_equal(o["idx"], [0, 2])
    np.testing.assert_array_equal(o["side"], [0, 1])
    # A margin column posts its margin out of the cash: equity stays init_cash.
    pf = alphax.from_orders(col(100, 110), col(N, N), init_cash=1000.0, init_position=-5.0,
                            account_type="margin", initial_margin=0.1)
    np.testing.assert_array_equal(pf.value().ravel(), [1000, 950])
    with pytest.raises(ValueError, match="cannot be held"):
        alphax.from_orders(col(100, 110), col(N, N), init_cash=10.0, init_position=-5.0,
                           account_type="margin", initial_margin=0.1)


def test_auto_cash_is_what_the_strategy_needed():
    pf = alphax.from_orders(col(10, 12, 8, 11), col(5, 5, -10, 20), init_cash="auto")
    np.testing.assert_array_equal(pf.cash().ravel(), [200, 140, 220, 0])
    # Two columns: each its own, or both the larger.
    close = np.array([[10.0, 10.0], [10.0, 10.0]])
    size = np.array([[1.0, 3.0], [N, N]])
    auto = alphax.from_orders(close, size, init_cash="auto")
    assert (auto.cash(0)[0], auto.cash(1)[0]) == (0.0, 0.0)
    aligned = alphax.from_orders(close, size, init_cash="autoalign")
    assert (aligned.cash(0)[0], aligned.cash(1)[0]) == (20.0, 0.0)


def test_auto_cash_with_shared_cash_is_the_group_total():
    # One account holding both columns: the first bar buys 1 + 3 shares at 10.
    close = np.array([[10.0, 10.0], [10.0, 10.0]])
    size = np.array([[1.0, 3.0], [N, N]])
    shared = alphax.from_orders(close, size, init_cash="auto", cash_sharing=True, group_lens=[2])
    assert shared.group_lens == (2,)
    assert shared.cash(group=0)[0] == 0.0
    np.testing.assert_array_equal(shared.init_cash, [40.0, 40.0])
    with pytest.raises(ValueError, match="do not depend on the cash"):
        alphax.from_orders(close, size, size_type="percent", init_cash="auto")


def test_negative_prices():
    close = col(18.0, -37.6, 10.0, 20.0)
    # A long future through the negative settlement: the loss in points.
    pf = alphax.from_orders(close, col(1, N, N, -1), init_cash=100_000.0, account_type="margin",
                            initial_margin=0.1, multiplier=1000.0, allow_negative_price=True)
    np.testing.assert_allclose(pf.value().ravel(), [100_000, 44_400, 92_000, 102_000])
    np.testing.assert_allclose(pf.trades()["pnl"], [2000.0])
    # A cash column paid to take the contract at -37.6, fees on its size.
    pf = alphax.from_orders(close, col(N, 1, N, -1), init_cash=100.0, fees=0.01,
                            allow_negative_price=True)
    np.testing.assert_allclose(pf.cash().ravel(), [100, 137.224, 137.224, 157.024])
    np.testing.assert_allclose(pf.trades()["ret"], [57.024 / 37.6])
    with pytest.raises(ValueError, match="greater than 0"):
        alphax.from_orders(close, col(N, 1, N, -1), init_cash=100.0)


def test_a_continuous_series_back_adjusts_at_the_roll():
    # Two contracts; the second is 5 above the first; roll on bar 2.
    prices = np.array([[100.0, 105.0], [101.0, 106.0], [102.0, 107.0], [N, 108.0]])
    held = alphax.futures.roll_by_expiry([2, 3], 4, before=1)
    np.testing.assert_array_equal(held, [0, 0, 1, 1])
    np.testing.assert_array_equal(alphax.futures.rolls(held), [False, False, True, False])
    np.testing.assert_array_equal(alphax.futures.continuous(prices, held), [105, 106, 107, 108])
    np.testing.assert_allclose(
        alphax.futures.continuous(prices, held, method="ratio"),
        [100 * 107 / 102, 101 * 107 / 102, 107, 108],
    )
    volume = np.array([[9.0, 1.0], [5.0, 6.0], [7.0, 2.0], [N, 3.0]])
    np.testing.assert_array_equal(alphax.futures.roll_by_volume(volume), [0, 1, 1, 1])
    params = alphax.futures.spec_params(["ES", "CL"])
    np.testing.assert_array_equal(params["multiplier"], [50.0, 1000.0])


def test_prices_in_the_account_currency():
    eur = col(100.0, 110.0)
    eurusd = np.array([1.1, 1.0])
    np.testing.assert_allclose(alphax.fx.convert(eur, eurusd).ravel(), [110.0, 110.0])
    np.testing.assert_allclose(alphax.fx.convert(col(15000.0), [150.0], invert=True), [[100.0]])
