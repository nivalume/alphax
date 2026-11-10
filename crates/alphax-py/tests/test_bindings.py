"""Tests for the Python bindings.

Layout, shapes, dtypes, error messages, thread determinism, and the fill,
stop and log behaviour of the signal and order paths, checked on small
hand-built cases.

Run from the repository root::

    python -m pip install -e 'crates/alphax-py[test]'
    pytest crates/alphax-py/tests
"""

from __future__ import annotations

import numpy as np
import pytest

import alphax

SEED = 20260919


def bars(n_rows: int, n_cols: int, seed: int = SEED):
    """A bar series with a consistent high >= close >= low."""
    rng = np.random.default_rng(seed)
    close = np.abs(rng.normal(100.0, 6.0, (n_rows, n_cols))) + 5.0
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.01, close.shape)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.01, close.shape)))
    opn = low + (high - low) * rng.random(close.shape)
    entries = rng.random(close.shape) < 0.15
    exits = rng.random(close.shape) < 0.15
    return opn, high, low, close, entries, exits


# --------------------------------------------------------------------------
# Shapes, targets, threads and errors
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "size_type", ["targetamount", "targetvalue", "targetpercent"]
)
def test_a_target_size_sets_where_the_signal_path_ends_up(size_type):
    """In the signal path an entry moves the position to the target: from
    flat, from the other side in one order, or from the same side where
    accumulation lets it add, and an exit closes. On a flat price the
    target's units are the same for all three types."""
    close = np.full((4, 1), 100.0)
    entries = np.array([[True], [False], [True], [False]])
    exits = np.array([[False], [True], [False], [False]])
    size = {"targetamount": 5.0, "targetvalue": 500.0, "targetpercent": 0.5}[size_type]
    pf = alphax.from_signals(
        close, entries, exits, size=size, size_type=size_type, init_cash=1000.0
    )
    np.testing.assert_allclose(pf.orders()["size"], [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(pf.orders()["side"], [0, 1, 0])


def test_threads_do_not_change_a_single_fill():
    """The reason columns are the parallel axis: they do not interact."""
    _, _, _, close, entries, exits = bars(300, 40, seed=4)
    one = alphax.from_signals(close, entries, exits, init_cash=1000.0, threads=1).orders()
    many = alphax.from_signals(close, entries, exits, init_cash=1000.0, threads=8).orders()
    assert len(one["col"]) > 100, "too few fills for this to prove anything"
    for field in one:
        np.testing.assert_array_equal(one[field], many[field], err_msg=field)


def test_a_c_ordered_array_is_refused_rather_than_transposed():
    """The trap the whole array module exists for.

    A square matrix is used deliberately: the transposed reading has the same
    shape, so nothing downstream could notice.
    """
    _, _, _, close, entries, exits = bars(40, 40)
    with pytest.raises(ValueError, match="C-ordered"):
        alphax._alphax.from_signals(
            np.ascontiguousarray(close),
            np.asfortranarray(entries),
            np.asfortranarray(exits),
        )


def test_the_wrapper_accepts_a_c_ordered_array_by_converting_it():
    _, _, _, close, entries, exits = bars(40, 4)
    c_order = np.ascontiguousarray(close)
    assert not c_order.flags["F_CONTIGUOUS"]
    a = alphax.from_signals(c_order, entries, exits, init_cash=1000.0).orders()
    b = alphax.from_signals(np.asfortranarray(close), entries, exits, init_cash=1000.0).orders()
    for field in a:
        np.testing.assert_array_equal(a[field], b[field], err_msg=field)


def test_a_non_contiguous_slice_is_refused():
    _, _, _, close, entries, exits = bars(40, 8)
    view = np.asfortranarray(close)[:, ::2]
    assert not view.flags["F_CONTIGUOUS"]
    with pytest.raises(ValueError):
        alphax._alphax.from_signals(
            view, np.asfortranarray(entries[:, ::2]), np.asfortranarray(exits[:, ::2])
        )


def test_a_stop_without_a_bar_range_is_refused():
    """A stop with no high and low can never fire, and would look like a run
    where the strategy simply never stopped out."""
    _, _, _, close, entries, exits = bars(60, 1)
    with pytest.raises(ValueError, match="high and low"):
        alphax.from_signals(close, entries, exits, sl_stop=0.02)
    with pytest.raises(ValueError, match="high and low"):
        alphax.from_signals(close, entries, exits, tp_stop=0.02)


def test_a_one_dimensional_series_is_one_column():
    _, _, _, close, entries, exits = bars(60, 1)
    flat = alphax.from_signals(close.ravel(), entries.ravel(), exits.ravel(), init_cash=100.0)
    shaped = alphax.from_signals(close, entries, exits, init_cash=100.0)
    assert flat.n_cols == shaped.n_cols == 1
    assert flat.n_rows == shaped.n_rows == 60
    np.testing.assert_array_equal(flat.orders()["idx"], shaped.orders()["idx"])


def test_a_mismatched_shape_names_both_shapes():
    _, _, _, close, entries, exits = bars(60, 4)
    with pytest.raises(ValueError, match="60x2 array against a 60x4 run"):
        alphax.from_signals(close, entries[:, :2], exits)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"direction": "sideways"}, "longonly"),
        ({"size_type": "Amount"}, "targetpercent"),
        ({"accumulate": "add"}, "addonly"),
        ({"trades": "exits"}, "'exit'"),
    ],
)
def test_an_unknown_enum_name_lists_the_alternatives(kwargs, message):
    _, _, _, close, entries, exits = bars(20, 1)
    with pytest.raises(ValueError, match=message):
        alphax.from_signals(close, entries, exits, **kwargs)


def test_parameters_broadcast_from_a_scalar_and_from_a_column_vector():
    """One value per column must mean the same as that value everywhere."""
    _, _, _, close, entries, exits = bars(80, 3)
    scalar = alphax.from_signals(close, entries, exits, init_cash=1000.0, fees=0.002)
    per_col = alphax.from_signals(
        close, entries, exits, init_cash=1000.0, fees=np.full(3, 0.002)
    )
    for field in scalar.orders():
        np.testing.assert_allclose(
            scalar.orders()[field], per_col.orders()[field], err_msg=field
        )


def test_per_column_parameters_actually_differ_per_column():
    """The previous test would pass if the vector were ignored entirely."""
    _, _, _, close, entries, exits = bars(80, 3)
    mixed = alphax.from_signals(
        close, entries, exits, init_cash=1000.0, fees=np.array([0.0, 0.01, 0.05])
    )
    orders = mixed.orders()
    by_col = [orders["fees"][orders["col"] == c].sum() for c in range(3)]
    assert by_col[0] == 0.0
    assert by_col[1] < by_col[2], "a higher fee rate produced no more fees"


def test_untraded_columns_are_masked_rather_than_ranked_first():
    """An untraded column's Sharpe is inf, which tops a naive ranking."""
    _, _, _, close, entries, exits = bars(80, 3)
    quiet = entries.copy()
    quiet[:, 1] = False  # column 1 never enters
    pf = alphax.from_signals(close, quiet, exits, init_cash=1000.0)
    sharpe = pf.sharpe_ratio()
    assert not np.isfinite(sharpe[1]), "an untraded column should not score finitely"
    masked = pf.traded_only(sharpe)
    assert np.isnan(masked[1])
    assert int(np.nanargmax(masked)) != 1


def test_records_come_back_as_arrays_with_the_right_dtypes():
    _, _, _, close, entries, exits = bars(120, 2)
    pf = alphax.from_signals(close, entries, exits, init_cash=1000.0, fees=0.001)
    orders, trades = pf.orders(), pf.trades()
    assert pf.n_fills == len(orders["col"])
    for field in ("col", "idx", "side"):
        assert orders[field].dtype == np.int64, field
    for field in ("size", "price", "fees"):
        assert orders[field].dtype == np.float64, field
    for field in ("col", "entry_idx", "exit_idx", "direction", "status", "parent_id"):
        assert trades[field].dtype == np.int64, field
    # Every trade belongs to a fill, so there cannot be more of them.
    assert len(trades["col"]) <= len(orders["col"])


def test_series_are_one_column_and_bounds_are_checked():
    _, _, _, close, entries, exits = bars(120, 3)
    pf = alphax.from_signals(close, entries, exits, init_cash=1000.0)
    assert pf.value(0).shape == (120,)
    assert pf.returns(2).shape == (120,)
    with pytest.raises(IndexError, match="outside 0..3"):
        pf.value(3)


def test_the_metrics_bundle_agrees_with_the_individual_accessors():
    """The bundle exists to save rebuilds, not to compute something else."""
    _, _, _, close, entries, exits = bars(150, 4)
    pf = alphax.from_signals(close, entries, exits, init_cash=1000.0, fees=0.001)
    bundle = pf.metrics()
    np.testing.assert_array_equal(bundle["total_return"], pf.total_return())
    np.testing.assert_array_equal(bundle["sharpe_ratio"], pf.sharpe_ratio())
    np.testing.assert_array_equal(bundle["max_drawdown"], pf.max_drawdown())
    np.testing.assert_array_equal(bundle["trade_count"], pf.trade_count())


def test_the_trade_view_changes_what_the_statistics_score():
    """Three views of the same fills, and they are not interchangeable."""
    _, _, _, close, entries, exits = bars(200, 1, seed=7)
    counts = {}
    for view in ("exit", "entry", "positions"):
        pf = alphax.from_signals(
            close, entries, exits, init_cash=1000.0, accumulate="both", size=1.0,
            trades=view,
        )
        counts[view] = int(pf.trade_count()[0])
    assert len(set(counts.values())) > 1, f"all three views agreed: {counts}"


def test_the_engine_does_not_write_through_a_borrowed_array():
    """Inputs are borrowed, not copied, so a stray write would reach into the
    caller's own array. The borrow is readonly on the Rust side; this checks
    the guarantee from the outside, where it is actually observable."""
    _, _, _, close, entries, exits = bars(60, 2)
    close = np.asfortranarray(close)
    before = close.copy()
    alphax.from_signals(close, entries, exits, init_cash=1000.0)
    # Not proof of borrowing on its own, but it does prove the engine does not
    # write through the borrow, which is the risk a readonly borrow takes.
    np.testing.assert_array_equal(close, before)


# --------------------------------------------------------------------------
# from_orders, and the log records
# --------------------------------------------------------------------------

def test_the_default_minimum_size_rejects_dust():
    """`min_size` defaults to 1e-8, so a size of 5e-9 is not an order.

    Observable, and invisible to randomly generated sizes, which are never dust.
    The log's `req_min_size` field records the default that was applied.
    """
    close = np.full((2, 1), 10.0)
    size = np.array([[5e-9], [0.0]])
    mine = alphax.from_orders(close, size, init_cash=100.0, log=True)
    assert mine.logs()["req_min_size"][0] == 1e-8
    assert mine.n_fills == 0, "dust is rejected by default"
    # With the guard off it fills, which is what makes the default a decision.
    assert alphax.from_orders(close, size, init_cash=100.0, min_size=0.0).n_fills == 1


def test_a_quiet_bar_gets_no_log_row_from_signals():
    """One row per order *considered*, not per bar.

    `from_orders` submits an order for every element, so it logs every element.
    `from_signals` only reaches the kernel when the signals resolved to a size,
    so a bar with no signal has nothing to explain — and a row per bar would
    bury the interesting ones under the bars where the strategy had no opinion.
    """
    close = np.array([[10.0], [11.0], [12.0], [11.0], [13.0]])
    entries = np.array([[True], [False], [False], [True], [False]])
    exits = np.array([[False], [False], [True], [False], [False]])
    signals = alphax.from_signals(close, entries, exits, init_cash=100.0, log=True)
    assert signals.n_logs == 3
    np.testing.assert_array_equal(signals.logs()["idx"], [0, 2, 3])

    orders = alphax.from_orders(close, np.ones((5, 1)), init_cash=100.0, log=True)
    assert orders.n_logs == 5, "from_orders considers every element"


def test_logging_is_off_by_default_and_changes_nothing_when_on():
    """Additional, never alternative. An investigation that changed the run
    would be investigating something else."""
    _, _, _, close, entries, exits = bars(120, 3)
    quiet = alphax.from_signals(close, entries, exits, init_cash=1000.0, fees=0.001)
    loud = alphax.from_signals(close, entries, exits, init_cash=1000.0, fees=0.001, log=True)
    assert quiet.n_logs == 0, "logging should be off unless asked for"
    assert loud.n_logs > 0
    for field in quiet.orders():
        np.testing.assert_array_equal(quiet.orders()[field], loud.orders()[field], err_msg=field)


def test_a_log_mask_logs_only_what_it_selects():
    """The useful shape on a sweep: one column under investigation."""
    _, _, _, close, entries, exits = bars(80, 4)
    mask = np.zeros(close.shape, dtype=bool)
    mask[:, 2] = True
    pf = alphax.from_signals(close, entries, exits, init_cash=1000.0, log=mask)
    assert pf.n_logs > 0, "nothing was logged, so this proves nothing"
    assert set(np.unique(pf.logs()["col"])) == {2}


def test_every_log_row_joins_to_its_own_fill():
    """`order_id` is an index into the order records, and the index is re-based
    when per-thread buffers merge. A mistake there names a real fill in another
    column — in range, wrong trade, and invisible to a length check."""
    _, _, _, close, entries, exits = bars(120, 6)
    for threads in (1, 4):
        pf = alphax.from_signals(
            close, entries, exits, init_cash=1000.0, log=True, threads=threads
        )
        logs, orders = pf.logs(), pf.orders()
        joined = logs["order_id"] >= 0
        assert joined.sum() > 3, "too few fills to prove anything"
        at = logs["order_id"][joined]
        np.testing.assert_array_equal(orders["col"][at], logs["col"][joined])
        np.testing.assert_array_equal(orders["idx"][at], logs["idx"][joined])
        # And an unfilled row points at nothing rather than at fill zero.
        unfilled = logs["res_status"][~joined]
        assert np.all(unfilled != 0), "a row with no order claims it filled"


def test_threads_do_not_change_the_logs():
    _, _, _, close, entries, exits = bars(200, 12)
    one = alphax.from_signals(close, entries, exits, init_cash=1000.0, log=True, threads=1)
    many = alphax.from_signals(close, entries, exits, init_cash=1000.0, log=True, threads=6)
    assert one.n_logs > 20, "too few rows to prove anything"
    a, b = one.logs(), many.logs()
    assert set(a) == set(b)
    for field in a:
        np.testing.assert_array_equal(
            np.asarray(a[field], dtype=float), np.asarray(b[field], dtype=float),
            err_msg=field,
        )


def test_reason_counts_tallies_one_column():
    close = np.full((6, 2), 100.0)
    size = np.array([[1.0, 0.0]] * 6)  # column 1 never orders anything
    pf = alphax.from_orders(close, size, init_cash=250.0, log=True)
    first = pf.reason_counts(0)
    assert sum(first.values()) == 6, "every considered order should be tallied"
    assert "filled" in first
    # Commonest first, which is what makes it a diagnostic rather than a dump.
    counts = list(first.values())
    assert counts == sorted(counts, reverse=True)
    assert set(pf.reason_counts(1)) == {"size is zero"}
    with pytest.raises(IndexError, match="outside 0..2"):
        pf.reason_counts(2)


def test_log_reasons_agree_with_the_status_fields():
    """The words are a lookup on the fields, so they must not disagree."""
    close = np.full((6, 1), 100.0)
    size = np.array([[1.0], [0.0], [1000.0], [1.0], [0.0], [-2.0]])
    pf = alphax.from_orders(close, size, init_cash=250.0, log=True, allow_partial=False)
    reasons, logs = pf.log_reasons(), pf.logs()
    assert len(reasons) == pf.n_logs == 6
    for i, reason in enumerate(reasons):
        filled = logs["res_status"][i] == 0
        assert (reason == "filled") == filled, f"row {i}: {reason!r}"
    assert "size is zero" in reasons


def test_from_orders_with_no_size_is_buy_and_hold():
    """The default size is `inf` and the default price is the close, so the
    whole balance goes in on the first bar."""
    close = np.array([[10.0], [11.0], [12.0], [13.0]])
    pf = alphax.from_orders(close, init_cash=100.0)
    assert pf.n_fills == 1
    np.testing.assert_array_equal(pf.orders()["idx"], [0])
    np.testing.assert_allclose(pf.total_return(), [0.3])


def test_order_mode_takes_target_sizes_as_from_orders_does():
    """A target size says where the position should end up, which is
    unambiguous per element: order mode reads ``size`` as from_orders does."""
    close = np.full((5, 1), 100.0)
    weights = np.full((5, 1), 0.5)
    pf = alphax.from_orders(close, weights, size_type="targetpercent", init_cash=1000.0)
    same = alphax.from_signals(
        close, size=weights, size_type="targetpercent", order_mode=True, direction="both",
        init_cash=1000.0,
    )
    assert pf.n_fills >= 1
    np.testing.assert_array_equal(same.orders()["size"], pf.orders()["size"])


def test_raise_reject_turns_a_refusal_into_an_exception():
    close = np.full((3, 1), 100.0)
    size = np.full((3, 1), 1000.0)  # far more than the cash allows
    refused = alphax.from_orders(close, size, init_cash=10.0, allow_partial=False, log=True)
    assert refused.n_fills == 0
    assert "filled" not in refused.log_reasons()
    with pytest.raises(ValueError, match="rejected"):
        alphax.from_orders(
            close, size, init_cash=10.0, allow_partial=False, raise_reject=True
        )
