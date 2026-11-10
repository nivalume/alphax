"""Stop exits without a simulation, and targets and order mode in
from_signals, through the Python API."""

import numpy as np
import pytest

import alphax
from alphax.signals import generate_ohlc_stop_exits

N = np.nan
OPEN = np.array([100.0, 99.0, 97.0, 93.0, 95.0])
HIGH = np.array([101.0, 100.0, 98.0, 96.0, 103.0])
LOW = np.array([99.0, 96.0, 94.0, 92.0, 95.0])
CLOSE = np.array([100.0, 97.0, 96.0, 95.0, 102.0])
ENTRY = np.array([True, False, False, False, False])


def test_stop_exits_from_the_bars_alone():
    exits, price, kind = generate_ohlc_stop_exits(
        ENTRY, CLOSE, open=OPEN, high=HIGH, low=LOW, sl_stop=0.0625,
    )
    np.testing.assert_array_equal(exits, [False, False, False, True, False])
    assert price[3] == 93.0 and kind[3] == 0
    assert np.isnan(price[0]) and kind[0] == -1
    _, _, kind = generate_ohlc_stop_exits(ENTRY, CLOSE, open=OPEN, td_stop=2.0)
    assert kind[2] == 3


def test_they_are_the_simulation_s_exits():
    rng = np.random.default_rng(4)
    close = 100.0 * np.cumprod(1.0 + 0.02 * rng.standard_normal((300, 3)), axis=0)
    open_ = close * (1.0 + 0.005 * rng.standard_normal((300, 3)))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    entries = rng.random((300, 3)) < 0.05
    stops = dict(sl_stop=0.03, tp_stop=0.05, tsl_stop=0.04)
    exits, price, _ = generate_ohlc_stop_exits(
        entries, close, open=open_, high=high, low=low, until_next=False,
        skip_until_exit=True, **stops,
    )
    pf = alphax.from_signals(
        close, entries, np.zeros_like(entries), open=open_, high=high, low=low,
        size=1.0, init_cash=1e9, **stops,
    )
    o = pf.orders()
    sells = o["side"] == 1
    expected = np.zeros_like(exits)
    expected[o["idx"][sells], o["col"][sells]] = True
    np.testing.assert_array_equal(exits, expected)
    np.testing.assert_allclose(price[o["idx"][sells], o["col"][sells]], o["price"][sells])


def test_a_target_entry_and_order_mode():
    close = np.array([100.0, 100.0, 100.0])
    entries = np.array([True, False, False])
    pf = alphax.from_signals(close, entries, size=0.5, size_type="targetpercent", init_cash=1000.0)
    np.testing.assert_array_equal(pf.orders()["size"], [5.0])
    size = np.array([[2.0], [N], [-1.0]])
    by_orders = alphax.from_orders(close, size, size_type="targetamount", init_cash=1000.0)
    by_signals = alphax.from_signals(
        close, size=size, size_type="targetamount", order_mode=True, direction="both",
        init_cash=1000.0,
    )
    for key in ["idx", "size", "price", "side"]:
        np.testing.assert_array_equal(by_signals.orders()[key], by_orders.orders()[key])


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="wait must be 1 or more"):
        generate_ohlc_stop_exits(ENTRY, CLOSE, sl_stop=0.1, wait=0)
    with pytest.raises(TypeError, match="entries must contain booleans"):
        generate_ohlc_stop_exits(ENTRY.astype(float), CLOSE)
