"""Python callbacks: from_order_func and from_signals' signal_func,
adjust_func, post_order_func and post_segment_func."""

import numpy as np
import pytest

import alphax


def _close(n=40, cols=3):
    r = np.arange(n)[:, None]
    c = np.arange(cols)[None, :]
    return 100.0 + 5.0 * np.sin(0.4 * r + c) + 0.1 * r


def _fills(pf):
    o = pf.orders()
    return list(zip(o["col"].tolist(), o["idx"].tolist(), o["size"].tolist(), o["price"].tolist()))


def test_an_order_func_matches_the_arrays_that_decide_the_same():
    close = _close()
    size = np.where(np.diff(close, axis=0, prepend=np.nan) > 0, 1.0, -1.0)
    size[0] = np.nan

    def order_func(ctx, k, state):
        col = ctx.from_col + k
        if ctx.row == 0:
            return None
        return 1.0 if close[ctx.row, col] > close[ctx.row - 1, col] else -1.0

    kwargs = dict(init_cash=500.0, fees=0.001, group_lens=[2, 1], cash_sharing=True)
    expected = alphax.from_orders(close, size, **kwargs)
    for row_wise in (False, True):
        pf = alphax.from_order_func(close, order_func, row_wise=row_wise, **kwargs)
        assert _fills(pf) == _fills(expected)
    assert expected.orders()["idx"].size > 20


def test_state_flex_orders_and_stopping():
    close = np.array([[10.0], [11.0], [12.0], [13.0]])

    def pre_group(group, from_col, to_col):
        return {"fills": 0, "seen": []}

    def two_legs(ctx, call, state):
        state["seen"].append((ctx.row, call))
        if call == 0:
            return (0, 2.0)
        if call == 1:
            return (0, {"size": -1.0, "fees": 0.01})
        return None

    def post_order(ctx, k, result, state):
        assert result["status"] in ("filled", "ignored", "rejected")
        if result["filled"]:
            state["fills"] += 1
        return "stop" if state["fills"] == 3 else None

    pf = alphax.from_order_func(
        close,
        two_legs,
        flexible=True,
        pre_group_func=pre_group,
        post_order_func=post_order,
        init_cash=1000.0,
        cash_earnings=1.0,
    )
    assert _fills(pf) == [(0, 0, 2.0, 10.0), (0, 0, 1.0, 10.0), (0, 1, 2.0, 11.0)]
    (state,) = pf.in_outputs
    assert state["fills"] == 3
    assert state["seen"] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    # The second leg paid its own fee rate.
    np.testing.assert_allclose(pf.orders()["fees"], [0.0, 0.1, 0.0])
    # Earned on every bar, after the stop too.
    np.testing.assert_allclose(pf.cash_flows("earning")[:, 0], [1.0] * 4)


def test_pre_segment_skips_and_post_segment_stops():
    close = _close(10, 1)

    def pre_segment(ctx, state):
        return "skip" if ctx.row % 2 else None

    def post_segment(ctx, state):
        return "stop" if ctx.row == 6 else None

    pf = alphax.from_order_func(
        close,
        lambda ctx, k, state: 1.0,
        pre_segment_func=pre_segment,
        post_segment_func=post_segment,
        init_cash=1e6,
    )
    np.testing.assert_array_equal(pf.orders()["idx"], [0, 2, 4, 6])


def test_row_wise_shares_state_across_groups_in_bar_order():
    close = _close(8, 4)
    taken = []

    def order_func(ctx, k, state):
        if sum(1 for row, _ in taken if row == ctx.row) < 2:
            taken.append((ctx.row, ctx.group))
            return 1.0
        return None

    alphax.from_order_func(close, order_func, row_wise=True, init_cash=1e6)
    assert taken == [(i, g) for i in range(8) for g in (0, 1)]


def test_a_callback_s_exception_is_what_the_caller_sees():
    close = _close(5, 1)

    class Boom(Exception):
        pass

    def order_func(ctx, k, state):
        if ctx.row == 3:
            raise Boom("from the callback")
        return None

    with pytest.raises(Boom, match="from the callback"):
        alphax.from_order_func(close, order_func)
    with pytest.raises(ValueError, match="None, a size or a dict"):
        alphax.from_order_func(close, lambda ctx, k, state: "lots")
    with pytest.raises(ValueError, match='no order field "qty"'):
        alphax.from_order_func(close, lambda ctx, k, state: {"qty": 1})
    with pytest.raises(ValueError, match="leverage"):
        alphax.from_order_func(close, lambda ctx, k, state: {"size": 1.0, "leverage": 3.0})
    with pytest.raises(TypeError, match="callable"):
        alphax.from_order_func(close, 3)


def test_signal_func_matches_the_arrays():
    close = _close()
    up = np.diff(close, axis=0, prepend=np.nan) > 0
    entries = up & (np.arange(len(close))[:, None] % 3 == 0)
    exits = ~up & (np.arange(len(close))[:, None] % 4 == 1)
    expected = alphax.from_signals(close, entries, exits, size=1.0, init_cash=500.0)

    def signal_func(ctx, k, state):
        col = ctx.from_col + k
        return (bool(entries[ctx.row, col]), bool(exits[ctx.row, col]), False, False)

    pf = alphax.from_signals(close, size=1.0, init_cash=500.0, signal_func=signal_func)
    assert _fills(pf) == _fills(expected)
    assert len(_fills(pf)) > 10


def test_risk_sizing_and_a_break_even_stop():
    close = np.array([100.0, 100.0, 103.0, 101.0, 99.0, 98.0])
    entries = np.array([False, True, False, False, False, False])
    sl = 0.05

    def signal_func(ctx, k, state):
        price = ctx.close[k]
        return {"size": 0.01 * ctx.value / (price * sl), "size_type": "amount"}

    def adjust_func(ctx, k, stops, state):
        if ctx.position[k] > 0 and ctx.close[k] >= stops["sl_price"] * 1.02:
            return {"sl_stop": 0.0}
        return None

    pf = alphax.from_signals(
        close,
        entries,
        high=close,
        low=close,
        sl_stop=sl,
        init_cash=10_000.0,
        signal_func=signal_func,
        adjust_func=adjust_func,
    )
    o = pf.orders()
    np.testing.assert_array_equal(o["idx"], [1, 4])
    np.testing.assert_allclose(o["size"], [20.0, 20.0])


def test_signal_callbacks_return_states_and_stop():
    close = _close(12, 2)
    entries = np.zeros_like(close, dtype=bool)
    entries[::2] = True
    exits = np.zeros_like(close, dtype=bool)
    exits[1::2] = True

    def pre_group(group, from_col, to_col):
        return [group]

    def post_order(ctx, k, result, state):
        state.append(ctx.row)
        return "stop" if len(state) == 4 else None

    pf = alphax.from_signals(
        close,
        entries,
        exits,
        size=1.0,
        init_cash=1e4,
        pre_group_func=pre_group,
        post_order_func=post_order,
    )
    assert pf.in_outputs == [[0, 0, 1, 2], [1, 0, 1, 2]]
    np.testing.assert_array_equal(pf.orders()["idx"], [0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="from_order_func"):
        alphax._alphax.from_signals(
            np.asfortranarray(close),
            np.asfortranarray(entries),
            np.asfortranarray(exits),
            row_wise=True,
        )
