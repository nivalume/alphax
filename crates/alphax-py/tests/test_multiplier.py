"""Contract multipliers through the Python API."""

import numpy as np
import pytest

import alphax

N = np.nan


def col(*values):
    return np.array(values, dtype=float).reshape(-1, 1)


def es(**kwargs):
    """One E-mini at 4000, 50 a point, 12000 margin a contract."""
    base = dict(
        size=col(1.0, N, N, -1.0), init_cash=20_000.0, account_type="margin",
        initial_margin=12_000.0, margin_basis="unit", multiplier=50.0,
    )
    base.update(kwargs)
    return alphax.from_orders(col(4000.0, 4010.0, 3990.0, 3990.0), **base)


def test_a_future_moves_by_the_multiplier_per_point():
    pf = es(log=True)
    orders = pf.orders()
    np.testing.assert_array_equal(orders["price"], [4000.0, 3990.0])
    np.testing.assert_array_equal(orders["multiplier"], [50.0, 50.0])
    np.testing.assert_allclose(np.ravel(pf.value()), [20_000.0, 20_500.0, 19_500.0, 19_500.0])
    np.testing.assert_allclose(np.ravel(pf.free_cash()), [8_000.0, 8_500.0, 7_500.0, 19_500.0])
    trades = pf.trades()
    np.testing.assert_allclose(trades["pnl"], [-500.0])
    np.testing.assert_allclose(trades["ret"], [-500.0 / 200_000.0])
    np.testing.assert_array_equal(trades["multiplier"], [50.0])
    logs = pf.logs()
    np.testing.assert_array_equal(logs["multiplier"], [50.0] * 4)
    np.testing.assert_array_equal(logs["res_price"][[0, 3]], [4000.0, 3990.0])
    np.testing.assert_array_equal(logs["val_price"], [4000.0, 4010.0, 3990.0, 3990.0])


def test_a_multiplier_is_the_same_run_in_larger_sizes():
    rng = np.random.default_rng(3)
    close = 100.0 * np.cumprod(1.0 + 0.02 * rng.standard_normal((80, 3)), axis=0)
    size = np.where(rng.random((80, 3)) < 0.3, rng.normal(0.0, 4.0, (80, 3)), N)
    kwargs = dict(
        init_cash=10_000.0, fees=0.001, slippage=0.001, account_type="margin",
        initial_margin=0.1, funding_rate=1e-4, cash_dividends=0.05,
    )
    k = np.array([50.0, 0.1, 1.0])
    multiplied = alphax.from_orders(close, size, multiplier=k, **kwargs)
    plain = alphax.from_orders(close, size * k, **kwargs)
    a, b = multiplied.orders(), plain.orders()
    np.testing.assert_allclose(a["size"] * k[a["col"]], b["size"])
    np.testing.assert_allclose(a["price"], b["price"])
    np.testing.assert_allclose(a["fees"], b["fees"])
    np.testing.assert_array_equal(a["multiplier"], k[a["col"]])
    for c in range(3):
        np.testing.assert_allclose(multiplied.value(c), plain.value(c))
    np.testing.assert_allclose(multiplied.cash_records()["amount"], plain.cash_records()["amount"])
    np.testing.assert_allclose(multiplied.trades()["pnl"], plain.trades()["pnl"])


def test_a_column_map_maps_the_multiplier_with_the_prices():
    close = np.array([[100.0, 10.0], [110.0, 11.0]])
    pf = alphax.from_orders(
        close, np.ones((2, 3)) * [[1.0], [N]], init_cash=1000.0, multiplier=[2.0, 5.0],
        column_map=[1, 0, 1],
    )
    np.testing.assert_array_equal(pf.orders()["multiplier"], [5.0, 2.0, 5.0])
    last = [pf.value(c)[-1] for c in range(3)]
    np.testing.assert_allclose(last, 1000.0 + np.array([5.0, 20.0, 5.0]))


def test_from_signals_takes_a_multiplier():
    pf = alphax.from_signals(
        np.array([10.0, 12.0, 11.0]), np.array([True, False, False]),
        np.array([False, True, False]), size=2.0, init_cash=1000.0, multiplier=10.0,
    )
    np.testing.assert_allclose(np.ravel(pf.value()), [1000.0, 1040.0, 1040.0])
    np.testing.assert_allclose(pf.trades()["pnl"], [40.0])


@pytest.mark.parametrize(
    "multiplier, match",
    [
        (0.0, "multiplier must be finite and greater than 0"),
        (-1.0, "multiplier must be finite and greater than 0"),
        (np.ones((4, 1)), "multiplier must be a number or one value per column"),
        ([1.0, 2.0], "per-column array has length 2, expected 1"),
    ],
)
def test_a_bad_multiplier_is_refused(multiplier, match):
    with pytest.raises((TypeError, ValueError), match=match):
        es(multiplier=multiplier)
