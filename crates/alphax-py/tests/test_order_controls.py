"""``cash_limit``, ``reject_prob`` and ``leverage`` through the Python API."""

import alphax
import numpy as np
import pytest


def test_cash_limit_caps_a_buy():
    pf = alphax.from_orders(np.array([10.0]), size=np.inf, cash_limit=30.0)
    np.testing.assert_array_equal(pf.orders()["size"], [3.0])


def test_reject_prob_is_reproducible_and_seeded():
    close = np.full((500, 4), 10.0, order="F")
    kwargs = dict(size=1.0, init_cash=1e9, reject_prob=0.4)
    first = alphax.from_orders(close, seed=1, **kwargs).orders()
    again = alphax.from_orders(close, seed=1, threads=3, **kwargs).orders()
    other = alphax.from_orders(close, seed=2, **kwargs).orders()
    np.testing.assert_array_equal(first["idx"], again["idx"])
    assert not np.array_equal(first["idx"], other["idx"])
    assert abs(len(first["idx"]) / 2000 - 0.6) < 0.05


def test_an_out_of_range_reject_prob_is_refused():
    with pytest.raises(ValueError, match="reject_prob"):
        alphax.from_orders(np.array([10.0]), size=1.0, reject_prob=1.5)


def test_a_levered_buy_borrows_and_value_stays_the_equity():
    close = np.array([[100.0], [120.0]])
    pf = alphax.from_orders(close, size=np.array([[15.0], [np.nan]]), init_cash=1000.0, leverage=2.0)
    orders = pf.orders()
    np.testing.assert_array_equal(orders["size"], [15.0])
    np.testing.assert_array_equal(orders["debt"], [500.0])
    np.testing.assert_array_equal(orders["locked_cash"], [1000.0])
    np.testing.assert_allclose(np.ravel(pf.cash()), [-500.0, -500.0])
    np.testing.assert_allclose(np.ravel(pf.value()), [1000.0, 1300.0])


def test_eager_leverage_puts_up_its_share_and_logs_the_request():
    pf = alphax.from_orders(
        np.array([100.0]), size=5.0, init_cash=1000.0,
        leverage=2.0, leverage_mode="eager", log=True,
    )
    np.testing.assert_array_equal(pf.orders()["debt"], [250.0])
    logs = pf.logs()
    np.testing.assert_array_equal(logs["req_leverage"], [2.0])
    np.testing.assert_array_equal(logs["req_leverage_mode"], [1])
    np.testing.assert_array_equal(logs["new_locked_cash"], [250.0])


def test_an_unlevered_run_has_no_balance_columns():
    pf = alphax.from_orders(np.array([100.0]), size=5.0, init_cash=1000.0)
    assert "debt" not in pf.orders()


def test_a_levered_signal_entry_buys_twice_the_cash():
    pf = alphax.from_signals(
        np.array([100.0, 110.0]), np.array([True, False]), np.array([False, True]),
        init_cash=1000.0, leverage=2.0,
    )
    np.testing.assert_array_equal(pf.orders()["size"], [20.0, 20.0])


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, np.nan])
def test_a_leverage_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="leverage"):
        alphax.from_orders(np.array([10.0]), size=1.0, leverage=bad)


def test_an_unknown_leverage_mode_is_refused():
    with pytest.raises(ValueError, match="leverage_mode"):
        alphax.from_orders(np.array([10.0]), size=1.0, leverage=2.0, leverage_mode="greedy")


def test_a_margin_long_posts_margin_and_settles_into_free_cash():
    close = np.array([[100.0], [110.0], [95.0]])
    pf = alphax.from_orders(
        close, size=np.array([[10.0], [np.nan], [-10.0]]), init_cash=1000.0,
        account_type="margin", initial_margin=0.2, log=True,
    )
    orders = pf.orders()
    np.testing.assert_array_equal(orders["debt"], [1000.0, 0.0])
    np.testing.assert_array_equal(orders["locked_cash"], [200.0, 0.0])
    np.testing.assert_array_equal(orders["margin"], [True, True])
    np.testing.assert_allclose(np.ravel(pf.value()), [1000.0, 1100.0, 950.0])
    np.testing.assert_allclose(np.ravel(pf.free_cash()), [800.0, 900.0, 950.0])
    logs = pf.logs()
    np.testing.assert_array_equal(logs["req_account_type"], [1, 1, 1])
    np.testing.assert_array_equal(logs["req_initial_margin"], [0.2, 0.2, 0.2])
    np.testing.assert_array_equal(logs["req_margin_basis"], [0, 0, 0])
    np.testing.assert_array_equal(logs["cash"], [1000.0, 1100.0, 950.0])


def test_a_spot_and_a_perpetual_share_one_account():
    close = np.array([[100.0, 100.0], [120.0, 120.0]])
    size = np.array([[5.0, -5.0], [np.nan, np.nan]])
    pf = alphax.from_orders(
        close, size=size, init_cash=1000.0, cash_sharing=True,
        account_type=["cash", "margin"], initial_margin=np.array([np.nan, 0.1]),
    )
    np.testing.assert_array_equal(pf.orders()["margin"], [False, True])
    np.testing.assert_allclose(np.ravel(pf.value()), [1000.0, 1000.0])
    np.testing.assert_allclose(np.ravel(pf.free_cash()), [450.0, 350.0])


def test_a_unit_margin_is_cash_per_contract():
    pf = alphax.from_signals(
        np.array([100.0, 400.0]), np.array([True, False]), np.array([False, False]),
        init_cash=1000.0, size=5.0, account_type="margin",
        initial_margin=15.0, margin_basis="unit",
    )
    np.testing.assert_array_equal(pf.orders()["locked_cash"], [75.0])


def test_a_cash_run_has_no_margin_column():
    pf = alphax.from_orders(np.array([100.0]), size=5.0, init_cash=1000.0, leverage=2.0)
    assert "margin" in pf.orders()
    assert not pf.orders()["margin"].any()
    assert "margin" not in alphax.from_orders(np.array([100.0]), size=5.0).orders()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(account_type="perp"), "account_type"),
        (dict(account_type=["margin"] * 3), "account_type has 3 entries for 2 columns"),
        (dict(account_type=[1, 2]), "account_type"),
        (dict(account_type=1), "account_type must be"),
        (dict(account_type="margin", margin_basis="lot"), "margin_basis"),
        (dict(account_type="margin", initial_margin=-0.1), "initial_margin"),
        (dict(account_type="margin", margin_basis="unit"), "initial_margin"),
    ],
)
def test_bad_margin_arguments_are_refused(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        alphax.from_orders(np.full((2, 2), 10.0), size=1.0, **kwargs)
