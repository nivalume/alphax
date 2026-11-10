"""Multi-resolution helpers, sessions and forced exits, and the percent
size types through the Python API."""

import numpy as np
import pytest

import alphax
from alphax.resolution import align, resample
from alphax.sessions import session_end, session_mask

N = np.nan


def minutes(start, n, step=15):
    return np.datetime64(start, "ns") + np.arange(n) * np.timedelta64(step, "m")


def test_coarse_values_land_where_they_become_known():
    fine = minutes("2024-01-02T10:00", 8)  # 10:00 .. 11:45
    hour_close = np.array(["2024-01-02T11:00", "2024-01-02T12:00"], dtype="datetime64[ns]")
    events = align(np.array([True, True]), hour_close, fine)
    # 11:00 is fine bar 4; noon is past the last.
    np.testing.assert_array_equal(events, [False] * 4 + [True] + [False] * 3)
    levels = align(np.array([0.01, 0.02]), hour_close, fine, fill=True)
    np.testing.assert_array_equal(np.isnan(levels[:4]), [True] * 4)
    np.testing.assert_array_equal(levels[4:], [0.01] * 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        align(np.array([True, True]), hour_close[::-1], fine)


def test_a_fine_series_reads_back_on_coarse_bars():
    fine = minutes("2024-01-02T10:15", 8)  # 10:15 .. 12:00
    hours = np.array(["2024-01-02T11:00", "2024-01-02T12:00"], dtype="datetime64[ns]")
    values = np.arange(8.0)
    np.testing.assert_array_equal(resample(values, fine, hours), [3.0, 7.0])
    np.testing.assert_array_equal(resample(values, fine, hours, how="sum"), [6.0, 22.0])
    r = np.full(8, 0.01)
    np.testing.assert_allclose(resample(r, fine, hours, how="compound"), [1.01**4 - 1] * 2)


def test_fine_bars_settle_which_stop_came_first():
    # Bought at 100, then one hour of four quarters: up to 103 first, then
    # down to 97. On the hour bar both the 2% take-profit and the 2%
    # stop-loss are inside its range, and the stop is assumed first; the
    # quarters show the take-profit was.
    fine = minutes("2024-01-02T10:00", 5)
    o = np.array([100, 100, 101, 102, 99], dtype=float)
    h = np.array([100, 101, 103, 102, 99], dtype=float)
    l = np.array([100, 100, 101, 99, 97], dtype=float)
    c = np.array([100, 101, 102, 99, 97], dtype=float)
    hour_close = fine[[0, 4]]
    entries = align(np.array([True, False]), hour_close, fine)
    stops = dict(sl_stop=0.02, tp_stop=0.02, size=1.0, init_cash=1000.0)
    pf = alphax.from_signals(c, entries, open=o, high=h, low=l, **stops)
    np.testing.assert_array_equal(pf.orders()["price"], [100.0, 102.0])
    # The hour bar of the same quarters: open 100, high 103, low 97, close 97.
    coarse = alphax.from_signals(
        np.array([100.0, 97.0]), np.array([True, False]),
        open=np.array([100.0, 100.0]), high=np.array([100.0, 103.0]),
        low=np.array([100.0, 97.0]), **stops,
    )
    np.testing.assert_array_equal(coarse.orders()["price"], [100.0, 98.0])


def test_a_session_in_the_exchange_s_time_zone():
    # New York: 14:30 UTC is 09:30 in winter (EST) and 10:30 in summer (EDT).
    winter = np.array(["2024-01-02T14:30", "2024-01-02T20:59", "2024-01-02T21:00"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(
        session_mask(winter, "09:30", "16:00", tz="America/New_York"), [True, True, False]
    )
    summer = np.array(["2024-07-02T13:30", "2024-07-02T19:59", "2024-07-02T20:00"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(
        session_mask(summer, "09:30", "16:00", tz="America/New_York"), [True, True, False]
    )
    # Saturday is not a session day.
    saturday = np.array(["2024-01-06T15:00"], dtype="datetime64[ns]")
    assert not session_mask(saturday, "09:30", "16:00", tz="America/New_York")[0]


def test_an_overnight_session_and_its_last_bar():
    # 18:00 to 17:00 UTC, Sunday to Thursday starts: Monday 16:00 belongs to
    # the session that began Sunday 18:00, and is its last bar here.
    times = np.array(
        ["2024-01-07T18:00", "2024-01-08T16:00", "2024-01-08T17:30", "2024-01-08T18:00"],
        dtype="datetime64[ns]",
    )
    mask = session_mask(times, "18:00", "17:00", weekdays=(6, 0, 1, 2, 3))
    np.testing.assert_array_equal(mask, [True, True, False, True])
    last = session_end(times, "18:00", "17:00", weekdays=(6, 0, 1, 2, 3))
    np.testing.assert_array_equal(last, [False, True, False, True])


def test_a_forced_exit_on_every_column():
    close = np.array([[100.0, 50.0], [101.0, 51.0], [102.0, 52.0], [103.0, 53.0]])
    entries = np.array([[True, True], [False, False], [False, False], [False, False]])
    force = np.array([False, False, True, False])
    pf = alphax.from_signals(close, entries, size=1.0, init_cash=1000.0, force_exit=force)
    o = pf.orders()
    np.testing.assert_array_equal(o["idx"], [0, 2, 0, 2])
    np.testing.assert_array_equal(o["side"], [0, 1, 0, 1])


def test_dropping_the_times_an_instrument_is_missing():
    panel = alphax.data.BarPanel(
        time=np.arange(3),
        bar_types=("a", "b"),
        open=np.ones((3, 2)),
        high=np.ones((3, 2)),
        low=np.ones((3, 2)),
        close=np.array([[1.0, 2.0], [1.0, N], [1.0, 2.0]]),
        volume=np.ones((3, 2)),
    )
    dropped = panel.drop_missing()
    np.testing.assert_array_equal(dropped.time, [0, 2])
    assert dropped.shape == (2, 2)


def test_the_percent_size_types():
    close = np.full((3, 1), 100.0)
    size = np.array([[50.0], [N], [N]])
    a = alphax.from_orders(close, size, size_type="percent100", init_cash=1000.0)
    b = alphax.from_orders(close, size / 100, size_type="percent", init_cash=1000.0)
    np.testing.assert_array_equal(a.orders()["size"], b.orders()["size"])
    c = alphax.from_orders(close, np.array([[0.3], [N], [N]]), size_type="valuepercent",
                           init_cash=1000.0)
    np.testing.assert_allclose(c.orders()["size"], [3.0])
    entries = np.array([True, False, False])
    d = alphax.from_signals(close[:, 0], entries, size=30.0, size_type="targetpercent100",
                            init_cash=1000.0)
    np.testing.assert_allclose(d.orders()["size"], [3.0])
