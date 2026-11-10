"""from_ago per bar, nextvalidopen, and an explicit call sequence through the
Python API."""

import numpy as np
import pytest

import alphax

N = np.nan


def test_nextvalidopen_skips_bars_without_a_price():
    close = np.array([100.0, N, N, 103.0, 104.0])
    open_ = np.array([100.0, N, N, 102.5, 103.5])
    entries = np.array([True, False, False, False, False])
    pf = alphax.from_signals(close, entries, open=open_, price="nextvalidopen", size=1.0)
    o = pf.orders()
    # Decided at bar 0's close, executed at the next open there is: bar 3.
    np.testing.assert_array_equal(o["idx"], [3])
    np.testing.assert_array_equal(o["price"], [102.5])
    # nextopen executes on bar 1, which has no data, at the close before it.
    pf = alphax.from_signals(close, entries, open=open_, price="nextopen", size=1.0)
    np.testing.assert_array_equal(pf.orders()["idx"], [1])
    np.testing.assert_array_equal(pf.orders()["price"], [100.0])


def test_from_ago_per_bar():
    close = np.array([100.0, 101.0, 102.0, 103.0])
    size = np.array([[1.0], [2.0], [3.0], [4.0]])
    pf = alphax.from_orders(close, size, from_ago=np.array([0, 1, 0, 5]), init_cash=1e6)
    o = pf.orders()
    np.testing.assert_array_equal(o["idx"], [0, 1, 2])
    np.testing.assert_array_equal(o["size"], [1.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="nonnegative"):
        alphax.from_orders(close, size, from_ago=np.array([0, -1, 0, 0]))
    with pytest.raises(ValueError, match="sets from_ago"):
        alphax.from_orders(close, size, price="nextopen", open=close, from_ago=2)


def test_an_explicit_call_sequence():
    close = np.full((1, 2), 100.0)
    size = np.ones((1, 2))
    kwargs = dict(init_cash=100.0, group_lens=[2], cash_sharing=True)
    pf = alphax.from_orders(close, size, call_seq=np.array([[1, 0]]), **kwargs)
    np.testing.assert_array_equal(pf.orders()["col"], [1])
    with pytest.raises(ValueError, match="appears twice"):
        alphax.from_orders(close, size, call_seq=np.array([[1, 1]]), **kwargs)
