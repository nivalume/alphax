"""Limit orders in from_signals through the Python API.

The bars, as in crates/alphax-sim/tests/limit_orders.rs::

    bar   open  high   low  close
     0     100   101    99    100
     1      99   100    96     97
     2      97    98    94     96
     3      93    96    92     95
     4      95   103    95    102
"""

import numpy as np
import pytest

import alphax

OPEN = np.array([100.0, 99.0, 97.0, 93.0, 95.0])
HIGH = np.array([101.0, 100.0, 98.0, 96.0, 103.0])
LOW = np.array([99.0, 96.0, 94.0, 92.0, 95.0])
CLOSE = np.array([100.0, 97.0, 96.0, 95.0, 102.0])
FIRST = np.array([True, False, False, False, False])
NO = np.zeros(5, dtype=bool)


def run(entries=FIRST, exits=NO, **kwargs):
    base = dict(
        open=OPEN, high=HIGH, low=LOW, size=1.0, init_cash=1000.0, slippage=0.01,
        order_type="limit",
    )
    base.update(kwargs)
    orders = alphax.from_signals(CLOSE, entries, exits, **base).orders()
    return list(zip(orders["idx"].tolist(), orders["price"].tolist()))


def test_a_buy_limit_rests_and_fills_at_its_price_without_slippage():
    assert run(limit_delta=5.0, delta_format="absolute") == [(2, 95.0)]
    assert run() == [(0, 100.0)]
    assert run(order_type="market") == [(0, 101.0)]


def test_a_gap_fills_at_the_open_the_limit_or_the_close():
    kwargs = dict(limit_delta=93.5, delta_format="target")
    assert run(**kwargs) == [(3, 93.0)]
    assert run(**kwargs, limit_order_price="hardlimit") == [(3, 93.5)]
    assert run(**kwargs, limit_order_price="close") == [(3, 95.0)]


def test_a_reversed_buy_is_a_breakout_entry():
    assert run(limit_delta=2.0, delta_format="absolute", limit_reverse=True) == [(4, 102.0)]


def test_time_in_force_and_expiry_count_bars():
    kwargs = dict(limit_delta=5.0, delta_format="absolute")
    assert run(**kwargs, limit_tif=1.0) == []
    assert run(**kwargs, limit_tif=2.0) == [(2, 95.0)]
    assert run(**kwargs, limit_expiry=2.0) == []


def test_an_order_placed_at_the_open_and_a_delayed_one():
    second = np.array([False, True, False, False, False])
    kwargs = dict(entries=second, price=-np.inf, limit_delta=2.0, delta_format="absolute")
    assert run(**kwargs) == [(1, 97.0)]
    assert run(**kwargs, limit_delay=True) == [(2, 97.0)]


def test_order_type_per_column():
    close = np.column_stack([CLOSE, CLOSE])
    pf = alphax.from_signals(
        close, np.column_stack([FIRST, FIRST]), np.column_stack([NO, NO]),
        open=np.column_stack([OPEN, OPEN]), high=np.column_stack([HIGH, HIGH]),
        low=np.column_stack([LOW, LOW]), size=1.0, init_cash=1000.0, slippage=0.01,
        order_type=["limit", "market"], limit_delta=5.0, delta_format="absolute",
    )
    orders = pf.orders()
    np.testing.assert_array_equal(orders["col"], [0, 1])
    np.testing.assert_array_equal(orders["idx"], [2, 0])
    np.testing.assert_allclose(orders["price"], [95.0, 101.0])


def test_the_limit_conflict_modes():
    twice = np.array([True, True, False, False, False])
    kwargs = dict(entries=twice, limit_delta=5.0, delta_format="absolute")
    assert run(**kwargs) == [(2, 95.0)]
    assert run(**kwargs, upon_adj_limit_conflict="cancelexecute") == [(3, 92.0)]
    assert run(**kwargs, upon_adj_limit_conflict="cancelignore") == []


def test_a_stop_and_a_reversal_on_one_bar():
    # Long 1 at 100, stopped at 94.5 on bar 2, where a short entry fires.
    sl = np.array([[0.055], [np.nan], [np.nan], [np.nan], [np.nan]])
    third = np.array([False, False, True, False, False])
    kwargs = dict(
        exits=third, direction="both", order_type="market", slippage=0.0, sl_stop=sl,
    )
    assert run(**kwargs) == [(0, 100.0), (2, 94.5)]
    assert run(**kwargs, upon_adj_stop_conflict="keepexecute") == [
        (0, 100.0), (2, 94.5), (2, 96.0),
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(order_type="stop"), "order_type must be 'market' or 'limit'"),
        (dict(order_type=["limit"] * 2), "order_type has 2 entries for 1 columns"),
        (dict(order_type=[1]), "order_type must be"),
        (dict(limit_order_price="open"), "limit_order_price must be"),
        (dict(delta_format="pct"), "delta_format must be"),
        (dict(limit_delta=-1.0), "limit_delta must be"),
        (dict(limit_tif=-1.0), "limit_tif and limit_expiry must be"),
        (dict(upon_opp_stop_conflict="keep"), "a conflict mode must be"),
    ],
)
def test_bad_limit_arguments_are_refused(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        run(**kwargs)
