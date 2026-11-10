"""Event studies, robustness statistics, labels, random signals and synthetic
prices through the Python API.

The engine tests the numbers. These test the shell: a 1-D input comes back
1-D, a C-ordered panel reads the same as an F-ordered one, seeds reproduce,
and errors arrive as Python exceptions that say what was wrong.
"""

import numpy as np
import pytest

import alphax
from alphax import events, labels, robust, signals, synthetic

# --------------------------------------------------------------------------
# Event studies
# --------------------------------------------------------------------------


def test_a_forward_path_is_relative_to_the_event_bar():
    close = np.array([1.0, 2.0, 4.0, 8.0])
    fired = np.array([False, True, False, False])
    study = events.forward_paths(close, fired, before=1, after=2)
    assert len(study) == 1
    np.testing.assert_array_equal(study.offsets, [-1, 0, 1, 2])
    np.testing.assert_allclose(study.paths, [[-0.5, 0.0, 1.0, 3.0]])
    assert list(study.col) == [0] and list(study.idx) == [1]
    np.testing.assert_allclose(study.mean_path(), [-0.5, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(study.median_path(), [-0.5, 0.0, 1.0, 3.0])
    assert study.hit_rate(2) == 1.0
    assert study.hit_rate(0) == 0.0
    assert np.isnan(study.hit_rate(5))


def test_an_event_near_the_end_is_padded_not_dropped():
    close = np.array([1.0, 2.0, 4.0])
    study = events.forward_paths(close, np.array([False, False, True]), after=2)
    assert study.paths.shape == (1, 3)
    assert study.paths[0, 0] == 0.0 and np.isnan(study.paths[0, 1:]).all()


def test_events_are_listed_column_by_column():
    close = np.asfortranarray(np.arange(1.0, 13.0).reshape(4, 3))
    fired = np.zeros((4, 3), dtype=bool)
    fired[0, 2] = fired[1, 0] = fired[2, 0] = True
    study = events.forward_paths(close, fired, after=1)
    assert list(study.col) == [0, 0, 2]
    assert list(study.idx) == [1, 2, 0]


def test_float_events_are_refused():
    with pytest.raises(TypeError, match="booleans"):
        events.forward_paths(np.ones(3), np.array([0.0, 1.0, np.nan]), after=1)


def test_events_must_match_close():
    with pytest.raises(ValueError, match="events"):
        events.forward_paths(np.ones((4, 2)), np.zeros((4, 3), dtype=bool), after=1)


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def returns_panel():
    rng = np.random.default_rng(3)
    return rng.normal(0.001, 0.01, (400, 5))


def test_probabilistic_sharpe_is_a_float_for_a_series_and_per_column_for_a_panel():
    r = returns_panel()
    panel = robust.probabilistic_sharpe(r)
    assert panel.shape == (5,)
    assert isinstance(robust.probabilistic_sharpe(r[:, 1]), float)
    assert robust.probabilistic_sharpe(r[:, 1]) == panel[1]
    assert np.all((panel >= 0.0) & (panel <= 1.0))
    # A higher bar is harder to clear.
    assert np.all(robust.probabilistic_sharpe(r, 0.1) < panel)


def test_deflating_against_a_sweep_lowers_the_probability():
    r = returns_panel()
    trials = np.nanmean(r, axis=0) / np.nanstd(r, axis=0, ddof=1)
    deflated = robust.deflated_sharpe(r, trials)
    assert np.all(deflated < robust.probabilistic_sharpe(r))
    assert robust.expected_max_sharpe(1, 0.5) == 0.0
    assert robust.expected_max_sharpe(100, 0.01) > robust.expected_max_sharpe(10, 0.01) > 0.0


def test_pbo_is_a_probability_and_checks_its_splits():
    r = returns_panel()
    assert 0.0 <= robust.pbo_cscv(r, n_splits=8) <= 1.0
    assert robust.pbo_cscv(r) == robust.pbo_cscv(np.ascontiguousarray(r), n_splits=16)
    with pytest.raises(ValueError, match="even"):
        robust.pbo_cscv(r, n_splits=7)
    with pytest.raises(ValueError, match="2 columns"):
        robust.pbo_cscv(r[:, :1], n_splits=4)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_labels_keep_the_shape_they_were_given():
    close = np.array([100.0, 101.0, 103.0, 99.0, 98.0])
    fwd = labels.fixed_horizon_returns(close, 1)
    assert fwd.shape == (5,)
    np.testing.assert_allclose(fwd[:4], close[1:] / close[:-1] - 1.0)
    assert np.isnan(fwd[4])
    trend = labels.trend_labels(close, 1, 0.015)
    assert trend.dtype == np.int8
    np.testing.assert_array_equal(trend, [0, 1, -1, 0, 0])


def test_a_c_ordered_panel_is_labelled_column_by_column():
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 1, (50, 3)), axis=0)
    c = labels.fixed_horizon_returns(np.ascontiguousarray(close), 3)
    for col in range(3):
        np.testing.assert_array_equal(c[:, col], labels.fixed_horizon_returns(close[:, col], 3))


def test_triple_barrier_says_which_barrier_and_when():
    close = np.array([100.0, 101.0, 103.0, 100.0, 96.0, 96.0])
    lab, at = labels.triple_barrier(close, up=0.02, down=0.03, max_hold=2)
    assert lab.dtype == np.int8 and at.dtype == np.int64
    # Bar 0 reaches +3% on bar 2; bar 2 falls 6.8% by bar 4; bar 1 meets
    # neither within two bars; the last bar has nothing after it.
    assert (lab[0], at[0]) == (1, 2)
    assert (lab[1], at[1]) == (0, 3)
    assert (lab[2], at[2]) == (-1, 4)
    assert (lab[5], at[5]) == (0, -1)


def test_pivots_are_confirmed_by_a_later_reversal():
    # 100 -> 90 confirms the peak at 100; 90 -> 100 the trough at 90; 110 ->
    # 95 the peak at 110; 95 -> 100 (+5.3%) the trough at 95. The last 100 is
    # a candidate peak nothing has reversed from yet.
    close = np.array([100.0, 90.0, 100.0, 110.0, 95.0, 100.0])
    piv = labels.pivot_labels(close, 0.05)
    assert piv.dtype == np.int8
    np.testing.assert_array_equal(piv, [-1, 1, 0, -1, 1, 0])


def test_bad_label_arguments_are_value_errors():
    with pytest.raises(ValueError, match="horizon"):
        labels.fixed_horizon_returns(np.ones(4), 0)
    with pytest.raises(ValueError, match="positive"):
        labels.triple_barrier(np.ones(4), up=0.0, down=0.1, max_hold=2)


# --------------------------------------------------------------------------
# Random signals
# --------------------------------------------------------------------------


def test_random_n_places_exactly_n_per_column():
    s = signals.random_n((100, 4), 7, seed=11)
    assert s.shape == (100, 4) and s.dtype == bool and s.flags.f_contiguous
    np.testing.assert_array_equal(s.sum(axis=0), [7, 7, 7, 7])


def test_a_seed_names_one_answer_and_a_column_ignores_its_neighbours():
    a = signals.random_by_prob((200, 3), 0.2, seed=5)
    np.testing.assert_array_equal(a, signals.random_by_prob((200, 3), 0.2, seed=5))
    assert not np.array_equal(a, signals.random_by_prob((200, 3), 0.2, seed=6))
    np.testing.assert_array_equal(a[:, 0], signals.random_by_prob(200, 0.2, seed=5))


def test_random_entries_and_exits_alternate():
    entries, exits = signals.random_entries_exits((80, 3), 5, seed=2)
    for col in range(3):
        events_ = np.where(entries[:, col], 1, 0) - np.where(exits[:, col], 1, 0)
        sequence = events_[events_ != 0]
        np.testing.assert_array_equal(sequence, [1, -1] * 5)


def test_an_impossible_count_is_refused():
    with pytest.raises(ValueError, match="cannot place"):
        signals.random_n(5, 6)
    with pytest.raises(ValueError, match="probability"):
        signals.random_by_prob(5, 1.5)


# --------------------------------------------------------------------------
# Synthetic prices
# --------------------------------------------------------------------------


def test_synthetic_paths_start_at_start_and_reproduce():
    g = synthetic.gbm((250, 4), seed=9)
    assert g.shape == (250, 4) and g.flags.f_contiguous
    np.testing.assert_array_equal(g[0], 100.0)
    assert np.all(g > 0.0)
    np.testing.assert_array_equal(g, synthetic.gbm((250, 4), seed=9))
    np.testing.assert_array_equal(g[:, 2], synthetic.gbm((250, 5), seed=9)[:, 2])
    w = synthetic.random_walk(30, start=5.0, step_std=0.5, seed=1)
    assert w.shape == (30,) and w[0] == 5.0


def test_a_flat_walk_stays_flat():
    np.testing.assert_array_equal(synthetic.random_walk((10, 2), step_std=0.0), 100.0)


def test_ohlc_brackets_the_close():
    close = synthetic.gbm((200, 2), seed=4)
    opn, high, low = synthetic.ohlc_from_close(close, spread=0.02, seed=4)
    assert opn.shape == high.shape == low.shape == close.shape
    np.testing.assert_array_equal(opn[1:], close[:-1])
    assert np.all(low <= np.minimum(opn, close)) and np.all(high >= np.maximum(opn, close))
    one_d = synthetic.ohlc_from_close(close[:, 0], spread=0.02, seed=4)
    np.testing.assert_array_equal(one_d[1], high[:, 0])


def test_bad_synthetic_arguments_are_value_errors():
    with pytest.raises(ValueError, match="sigma"):
        synthetic.gbm(10, sigma=-1.0)
    with pytest.raises(ValueError, match="spread"):
        synthetic.ohlc_from_close(np.ones(3), spread=1.0)
    with pytest.raises(ValueError, match="shape"):
        synthetic.gbm((2, 3, 4))


def test_the_research_modules_are_on_the_package():
    for name in ("events", "labels", "robust", "signals", "synthetic"):
        assert name in alphax.__all__
        assert getattr(alphax, name).__all__
