"""Price options and the extended stops through the Python API.

The bars mirror crates/alphax-sim/tests/{price_options,stop_extensions}.rs.
"""

import numpy as np
import pytest

import alphax

N = np.nan
OPEN = np.array([100.0, 99.0, 97.0, 93.0, 95.0])
HIGH = np.array([101.0, 100.0, 98.0, 96.0, 103.0])
LOW = np.array([99.0, 96.0, 94.0, 92.0, 95.0])
CLOSE = np.array([100.0, 97.0, 96.0, 95.0, 102.0])
ENTRY = np.array([True, False, False, False, False])
NO = np.zeros(5, dtype=bool)

UP_OPEN = np.array([100.0, 101.0, 107.0, 112.0])
UP_HIGH = np.array([100.0, 110.0, 108.0, 115.0])
UP_LOW = np.array([100.0, 100.0, 104.0, 111.0])
UP_CLOSE = np.array([100.0, 108.0, 105.0, 114.0])


def fills(pf):
    o = pf.orders()
    return list(zip(o["idx"].tolist(), o["price"].tolist(), o["size"].tolist()))


def orders(size, **kwargs):
    return alphax.from_orders(
        CLOSE, np.array(size).reshape(-1, 1), open=OPEN, high=HIGH, low=LOW, init_cash=1000.0, **kwargs
    )


def signals(entries=ENTRY, exits=NO, **kwargs):
    return alphax.from_signals(
        CLOSE, entries, exits, open=OPEN, high=HIGH, low=LOW, size=1.0, init_cash=1000.0,
        **kwargs,
    )


def up(**kwargs):
    base = dict(open=UP_OPEN, high=UP_HIGH, low=UP_LOW, size=1.0, init_cash=1000.0)
    base.update(kwargs)
    return alphax.from_signals(UP_CLOSE, ENTRY[:4], NO[:4], **base)


def test_named_prices_and_the_next_open():
    size = [1.0, N, -1.0, N, N]
    assert fills(orders(size, price="open")) == [(0, 100.0, 1.0), (2, 97.0, 1.0)]
    assert fills(orders(size, price="nextopen")) == [(1, 99.0, 1.0), (3, 93.0, 1.0)]
    assert fills(orders(size, price="nextclose")) == [(1, 97.0, 1.0), (3, 95.0, 1.0)]
    exits = np.array([False, False, True, False, False])
    assert fills(signals(exits=exits, price="nextopen")) == [(1, 99.0, 1.0), (3, 93.0, 1.0)]
    assert fills(orders(size, from_ago=2)) == [(2, 96.0, 1.0), (4, 102.0, 1.0)]


def test_a_buy_takes_the_ask_and_a_sell_the_bid():
    bid, ask = CLOSE - 0.5, CLOSE + 0.5
    pf = orders([1.0, N, -1.0, N, N], bid_price=bid, ask_price=ask)
    assert fills(pf) == [(0, 100.5, 1.0), (2, 95.5, 1.0)]


def test_a_price_outside_the_bar():
    price = np.array([[120.0], [N], [N], [N], [N]])
    size = [1.0, N, N, N, N]
    assert fills(orders(size, price=price, price_area_vio_mode="cap")) == [(0, 101.0, 1.0)]
    with pytest.raises(ValueError, match="outside the bar's range"):
        orders(size, price=price, price_area_vio_mode="error")


def test_a_trailing_and_a_time_stop():
    assert fills(up(tsl_stop=0.05)) == [(0, 100.0, 1.0), (2, 104.5, 1.0)]
    assert fills(up(tsl_stop=0.05, tsl_th=0.2)) == [(0, 100.0, 1.0)]
    assert fills(up(td_stop=2.0)) == [(0, 100.0, 1.0), (2, 107.0, 1.0)]
    assert fills(up(dt_stop=3.0)) == [(0, 100.0, 1.0), (3, 112.0, 1.0)]


def test_a_take_profit_ladder():
    steps = np.array([0.05, 0.1])
    expected = [(0, 100.0, 2.0), (1, 105.0, 1.0), (3, 112.0, 1.0)]
    assert fills(up(size=2.0, tp_stop=steps, stop_ladder="uniform")) == expected
    # Per column: an (n_steps, n_cols) array.
    assert fills(up(size=2.0, tp_stop=steps[:, None], stop_ladder="uniform")) == expected


def test_a_stop_limit_order_rests_after_a_gap():
    kwargs = dict(sl_stop=0.0625, stop_order_type="limit", stop_limit_delta=1 / 128)
    assert fills(signals(**kwargs)) == [(0, 100.0, 1.0), (4, 95.0, 1.0)]
    assert fills(signals(sl_stop=0.0625)) == [(0, 100.0, 1.0), (3, 93.0, 1.0)]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(price="tomorrow"), "price must be a number, an array, or 'open'"),
        (dict(price="nextopen", from_ago=2), "sets from_ago"),
        (dict(price_area_vio_mode="clip"), "price_area_vio_mode must be"),
        (dict(stop_ladder="yes"), "stop_ladder must be"),
        (dict(stop_order_type="stop"), "order_type must be 'market' or 'limit'"),
        (dict(td_stop=-1.0), "td_stop must be"),
        (
            dict(sl_stop=np.ones((2, 3)), stop_ladder="uniform"),
            "one column of steps per column",
        ),
    ],
)
def test_bad_price_and_stop_arguments_are_refused(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        signals(**kwargs)
