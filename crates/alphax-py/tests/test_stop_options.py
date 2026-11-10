"""Stop distances in delta_format, adaptive and trailing ladders, time stops
and order expiries given as time, and stop exits with ladders and stop-limit
orders, through the Python API."""

import numpy as np
import pytest

import alphax
from alphax.signals import generate_ohlc_stop_exits

OPEN = np.array([100.0, 101.0, 107.0, 112.0])
HIGH = np.array([100.0, 110.0, 108.0, 115.0])
LOW = np.array([100.0, 100.0, 104.0, 111.0])
CLOSE = np.array([100.0, 108.0, 105.0, 114.0])
ENTRY = np.array([True, False, False, False])
DAYS = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]")


def run(**kwargs):
    kwargs = dict(dict(open=OPEN, high=HIGH, low=LOW, size=1.0, init_cash=1000.0), **kwargs)
    o = alphax.from_signals(CLOSE, ENTRY, **kwargs).orders()
    return list(zip(o["idx"].tolist(), o["price"].tolist(), o["size"].tolist()))


def test_stop_distances_follow_delta_format():
    for fmt, tp in [("absolute", 8.0), ("percent100", 8.0), ("target", 108.0)]:
        assert run(tp_stop=tp, delta_format=fmt) == [(0, 100.0, 1.0), (1, 108.0, 1.0)], fmt


def test_a_trailing_ladder_and_the_adaptive_modes_are_accepted():
    assert run(size=2.0, tsl_stop=np.array([0.02, 0.05]), stop_ladder="uniform") == [
        (0, 100.0, 2.0),
        (2, 107.0, 1.0),
    ]
    out = run(size=3.0, tp_stop=np.array([0.05, 0.08, 0.12]), stop_ladder="adaptuniform")
    assert [s for _, _, s in out] == [3.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="stop_ladder must be"):
        run(tp_stop=0.1, stop_ladder="dynamic")


def test_time_stops_as_time():
    # Two days after the entry bar's time is bar 2: its open.
    rows = run(td_stop=2.0)
    assert run(td_stop=np.timedelta64(2, "D"), index=DAYS) == rows == [(0, 100.0, 1.0), (2, 107.0, 1.0)]
    # A moment: the first bar at or after it.
    assert run(dt_stop=np.datetime64("2024-01-03T12:00"), index=DAYS) == [(0, 100.0, 1.0), (3, 112.0, 1.0)]
    with pytest.raises(ValueError, match="needs the bars' times"):
        run(td_stop=np.timedelta64(2, "D"))
    with pytest.raises(ValueError, match="time_delta_format"):
        run(td_stop=np.timedelta64(2, "D"), time_delta_format="rows", index=DAYS)


def test_periods_and_times_of_day_need_pandas():
    pd = pytest.importorskip("pandas")
    index = pd.date_range("2024-01-29", periods=4, freq="D")  # Mon..Thu, month end on Wed
    # The end of the month the entry is in: 2024-01-31, bar 2.
    assert run(dt_stop="ME", index=index) == [(0, 100.0, 1.0), (2, 107.0, 1.0)]
    assert run(td_stop="2D", index=index) == [(0, 100.0, 1.0), (2, 107.0, 1.0)]
    hours = pd.date_range("2024-01-29 16:00", periods=4, freq="h", tz="UTC")
    assert run(dt_stop="18:00", index=hours) == [(0, 100.0, 1.0), (2, 107.0, 1.0)]


def test_a_limit_order_expires_in_time():
    # A buy limit 2% under bar 0's close, 98, is first reached by bar 2's
    # low. Good for one day, it is gone by then; for three, it fills.
    close = np.array([100.0, 100.0, 95.0, 95.0])
    low = np.array([100.0, 99.0, 94.0, 94.0])
    kwargs = dict(order_type="limit", limit_delta=0.02, low=low, high=close, open=close, size=1.0)
    filled = alphax.from_signals(close, ENTRY, **kwargs).orders()["idx"]
    np.testing.assert_array_equal(filled, [2])
    gone = alphax.from_signals(close, ENTRY, limit_tif=np.timedelta64(1, "D"), index=DAYS, **kwargs)
    assert len(gone.orders()["idx"]) == 0
    kept = alphax.from_signals(close, ENTRY, limit_tif=np.timedelta64(3, "D"), index=DAYS, **kwargs)
    np.testing.assert_array_equal(kept.orders()["idx"], [2])


def test_stop_exit_type_is_upon_stop_exit():
    a = run(sl_stop=0.01, tp_stop=0.05, upon_stop_exit="reverse", direction="both")
    b = run(sl_stop=0.01, tp_stop=0.05, stop_exit_type="reverse", direction="both")
    assert a == b
    with pytest.raises(ValueError, match="pass one of them"):
        run(stop_exit_type="reverse", upon_stop_exit="close")


def test_generated_exits_with_a_ladder_and_a_stop_limit():
    out = {}
    exits, price, kind = generate_ohlc_stop_exits(
        ENTRY, CLOSE, open=OPEN, high=HIGH, low=LOW, tp_stop=np.array([0.05, 0.08]),
        stop_ladder="uniform", out=out,
    )
    np.testing.assert_array_equal(exits, [False, True, True, False])
    np.testing.assert_array_equal(price[1:3], [105.0, 108.0])
    np.testing.assert_array_equal(out["size"][1:3], [0.5, 0.5])
    exits, price, _ = generate_ohlc_stop_exits(
        ENTRY, CLOSE, open=OPEN, high=HIGH, low=LOW, td_stop=2.0,
    )
    assert exits[2] and price[2] == 107.0
    exits, _, _ = generate_ohlc_stop_exits(
        ENTRY, CLOSE, open=OPEN, high=HIGH, low=LOW, tp_stop=0.05,
        stop_order_type="limit", stop_limit_delta=0.0,
    )
    assert exits[1]
