"""The from_signals stop modes, through the Python API.

The engine's own tests pin each mode against golden outputs. These pin the
binding: that each string reaches its own variant, which a run can only show
by a fill landing at a different price or a different bar. Every price below
is worked out by hand from the bars, so a mode that silently fell back to the
default would fail by a visible amount rather than a rounding error.
"""

import numpy as np
import pytest

import alphax

# One long entry on bar 0 at a close of 100. A 5% stop-loss anchored there
# sits at 95: bar 1 stays above it, bar 2 opens above it and trades through
# it, closing at 97.
OPEN = np.array([100.0, 100.0, 99.0])
HIGH = np.array([101.0, 101.0, 101.0])
LOW = np.array([99.0, 99.0, 90.0])
CLOSE = np.array([100.0, 100.0, 97.0])
ENTRIES = np.array([True, False, False])
EXITS = np.zeros(3, dtype=bool)


def stopped(**kwargs):
    return alphax.from_signals(
        CLOSE, ENTRIES, EXITS, open=OPEN, high=HIGH, low=LOW,
        init_cash=1000.0, size=1.0, sl_stop=0.05, **kwargs,
    )


def exit_price(pf):
    orders = pf.orders()
    assert list(orders["idx"]) == [0, 2], "expected the entry and one stop exit"
    return orders["price"][1]


def test_a_stop_limit_fills_at_the_level_without_slippage():
    assert exit_price(stopped(slippage=0.01)) == pytest.approx(95.0)
    assert exit_price(stopped(slippage=0.01, stop_exit_price="stoplimit")) == pytest.approx(95.0)


def test_a_stop_market_pays_slippage_on_the_level():
    pf = stopped(slippage=0.01, stop_exit_price="stopmarket")
    assert exit_price(pf) == pytest.approx(95.0 * 0.99)


def test_a_stop_can_exit_at_the_close_instead():
    assert exit_price(stopped(stop_exit_price="close")) == pytest.approx(97.0)


def test_a_stop_anchored_at_the_fill_moves_with_slippage():
    # The entry fills at 101 with 1% slippage. Anchored at the close the stop
    # stays at 95; anchored at the fill it is 5% under 101.
    at_close = stopped(slippage=0.01)
    at_fill = stopped(slippage=0.01, stop_entry_price="fillprice")
    assert at_fill.orders()["price"][0] == pytest.approx(101.0)
    assert exit_price(at_close) == pytest.approx(95.0)
    assert exit_price(at_fill) == pytest.approx(101.0 * 0.95)


def test_a_reversing_stop_leaves_a_short_and_a_closing_one_leaves_nothing():
    closed = stopped(direction="both")
    reversed_ = stopped(direction="both", upon_stop_exit="reverse")
    assert closed.assets(0)[-1] == 0.0
    assert reversed_.assets(0)[-1] < 0.0
    assert stopped(direction="both", upon_stop_exit="close").assets(0)[-1] == 0.0


# Two entries, on bar 0 at 100 and bar 2 at 110, adding to the position.
# The original stop is at 95; re-anchored on the add it is at 104.5. Bar 3
# trades down to 100, through the moved stop only; bar 4 gaps to 96 and
# trades down to 80, through the original one.
ADD_OPEN = np.array([100.0, 100.0, 109.0, 108.0, 96.0])
ADD_HIGH = np.array([101.0, 101.0, 111.0, 109.0, 97.0])
ADD_LOW = np.array([99.0, 99.0, 109.0, 100.0, 80.0])
ADD_CLOSE = np.array([100.0, 100.0, 110.0, 101.0, 90.0])
ADD_ENTRIES = np.array([True, False, True, False, False])


def added(sl_on_add, **kwargs):
    sl = np.full((5, 1), 0.05)
    sl[2, 0] = sl_on_add
    pf = alphax.from_signals(
        ADD_CLOSE, ADD_ENTRIES, np.zeros(5, dtype=bool),
        open=ADD_OPEN, high=ADD_HIGH, low=ADD_LOW,
        init_cash=1000.0, size=1.0, accumulate="addonly", sl_stop=sl, **kwargs,
    )
    orders = pf.orders()
    sells = orders["side"] == 1
    return list(orders["idx"][sells]), list(orders["price"][sells])


def test_the_default_update_moves_the_stop_to_the_add():
    assert added(0.05) == ([3], [pytest.approx(104.5)])
    assert added(0.05, upon_stop_update="override") == ([3], [pytest.approx(104.5)])


def test_keep_leaves_the_original_stop_in_place():
    assert added(0.05, upon_stop_update="keep") == ([4], [pytest.approx(95.0)])


def test_a_nan_stop_on_the_add_keeps_or_clears_by_mode():
    assert added(np.nan, upon_stop_update="override") == ([4], [pytest.approx(95.0)])
    assert added(np.nan, upon_stop_update="overridenan") == ([], [])


@pytest.mark.parametrize(
    "name,value,alternative",
    [
        ("stop_entry_price", "fill", "fillprice"),
        ("stop_exit_price", "market", "stopmarket"),
        ("upon_stop_exit", "flip", "reverse"),
        ("upon_stop_update", "nan", "overridenan"),
    ],
)
def test_an_unknown_stop_mode_names_the_alternatives(name, value, alternative):
    with pytest.raises(ValueError, match=alternative):
        stopped(**{name: value})
