"""Tests for alphax.validation, loaded from its file so no Rust build is needed."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "python" / "alphax" / "validation.py"
_spec = importlib.util.spec_from_file_location("alphax_validation", _PATH)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def _assert_disjoint(split, n):
    assert split.train.dtype == np.int64 and split.test.dtype == np.int64
    assert np.intersect1d(split.train, split.test).size == 0
    for rows in (split.train, split.test):
        assert rows.size > 0
        assert rows.min() >= 0 and rows.max() < n
        assert np.all(np.diff(rows) > 0)


def test_rolling_windows():
    splits = validation.rolling(n=20, train=5, test=3)
    assert [(s.train[0], s.train[-1] + 1, s.test[0], s.test[-1] + 1) for s in splits] == [
        (0, 5, 5, 8),
        (3, 8, 8, 11),
        (6, 11, 11, 14),
        (9, 14, 14, 17),
        (12, 17, 17, 20),
    ]
    for s in splits:
        _assert_disjoint(s, 20)
        assert s.train.size == 5 and s.test.size == 3
        assert s.train.max() < s.test.min()


def test_rolling_custom_step_and_short_tail_dropped():
    splits = validation.rolling(n=21, train=4, test=2, step=5)
    assert [s.train[0] for s in splits] == [0, 5, 10, 15]
    assert splits[-1].test.tolist() == [19, 20]
    # A tail too short for a whole test window is not tested.
    splits = validation.rolling(n=10, train=4, test=4)
    assert len(splits) == 1 and splits[0].test.tolist() == [4, 5, 6, 7]


def test_expanding_train_starts_at_zero():
    splits = validation.expanding(n=20, min_train=6, test=4)
    assert [(s.train.size, s.test[0], s.test[-1] + 1) for s in splits] == [
        (6, 6, 10),
        (10, 10, 14),
        (14, 14, 18),
    ]
    for s in splits:
        _assert_disjoint(s, 20)
        assert s.train[0] == 0
        np.testing.assert_array_equal(s.train, np.arange(s.test[0]))
    splits = validation.expanding(n=10, min_train=3, test=2, step=1)
    assert [s.train.size for s in splits] == [3, 4, 5, 6, 7, 8]


def test_purged_kfold_no_purge_is_plain_kfold():
    splits = validation.purged_kfold(n=10, k=3)
    assert [s.test.tolist() for s in splits] == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for s in splits:
        _assert_disjoint(s, 10)
        np.testing.assert_array_equal(np.union1d(s.train, s.test), np.arange(10))


def test_purged_kfold_purge_and_embargo_remove_exact_rows():
    n, purge, embargo = 30, 2, 3
    splits = validation.purged_kfold(n=n, k=3, purge=purge, embargo=embargo)
    # Middle fold: test [10, 20); purge drops 8, 9; embargo drops 20, 21, 22.
    middle = splits[1]
    assert middle.test.tolist() == list(range(10, 20))
    expected = [r for r in range(n) if r < 8 or r >= 23]
    assert middle.train.tolist() == expected
    # First fold: nothing before it to purge; embargo drops 10, 11, 12.
    assert splits[0].train.tolist() == list(range(13, 30))
    # Last fold: nothing after it to embargo; purge drops 18, 19.
    assert splits[2].train.tolist() == list(range(0, 18))
    for s in splits:
        _assert_disjoint(s, n)
        lo, hi = s.test[0], s.test[-1] + 1
        removed = np.setdiff1d(np.arange(n), np.union1d(s.train, s.test))
        expected_removed = [r for r in range(n) if lo - purge <= r < lo or hi <= r < hi + embargo]
        assert removed.tolist() == expected_removed


def test_purged_kfold_test_folds_cover_series_once():
    splits = validation.purged_kfold(n=17, k=5, purge=1, embargo=1)
    tests = np.concatenate([s.test for s in splits])
    np.testing.assert_array_equal(tests, np.arange(17))


def test_walk_forward_selects_argmax_on_train_and_reports_its_test_score():
    splits = validation.rolling(n=12, train=4, test=2)
    calls = []

    def fn(rows):
        calls.append(rows)
        # On train rows "b" wins; on test rows "a" would win. The test score
        # reported must be "b"'s, not the best test score.
        if rows.size == 4:
            return {"a": 1.0, "b": 2.0 + rows[0], "c": float("nan")}
        return {"a": 100.0, "b": -float(rows[0]), "c": 50.0}

    results = validation.walk_forward(fn, 12, splits)
    assert len(results) == len(splits)
    for i, (r, s) in enumerate(zip(results, splits)):
        assert r == {
            "split": i,
            "chosen": "b",
            "train_score": 2.0 + s.train[0],
            "test_score": -float(s.test[0]),
        }
    assert len(calls) == 2 * len(splits)
    np.testing.assert_array_equal(calls[0], splits[0].train)
    np.testing.assert_array_equal(calls[1], splits[0].test)


def test_walk_forward_custom_select_and_ties():
    splits = validation.expanding(n=8, min_train=4, test=2)

    def fn(rows):
        return {"x": 1.0, "y": 1.0, "z": 0.5}

    assert {r["chosen"] for r in validation.walk_forward(fn, 8, splits)} == {"x"}
    lowest = validation.walk_forward(fn, 8, splits, select=lambda s: min(s, key=s.get))
    assert [r["chosen"] for r in lowest] == ["z", "z"]


def test_walk_forward_errors():
    splits = validation.rolling(n=10, train=4, test=2)
    good = lambda rows: {"a": 1.0}  # noqa: E731
    with pytest.raises(ValueError, match="empty"):
        validation.walk_forward(good, 10, [])
    with pytest.raises(ValueError, match="outside"):
        validation.walk_forward(good, 8, splits)
    overlap = validation.Split(np.arange(4), np.arange(3, 5))
    with pytest.raises(ValueError, match="both train and test"):
        validation.walk_forward(good, 10, [overlap])
    with pytest.raises(ValueError, match="no scores"):
        validation.walk_forward(lambda rows: {}, 10, splits)
    with pytest.raises(ValueError, match="NaN"):
        validation.walk_forward(lambda rows: {"a": math.nan}, 10, splits)
    with pytest.raises(ValueError, match="infinite"):
        validation.walk_forward(lambda rows: {"a": math.inf, "b": -math.inf}, 10, splits)
    with pytest.raises(ValueError, match="not a train score key"):
        validation.walk_forward(good, 10, splits, select=lambda s: "zz")
    missing = lambda rows: {"a": 1.0} if rows.size == 4 else {"b": 1.0}  # noqa: E731
    with pytest.raises(ValueError, match="did not score"):
        validation.walk_forward(missing, 10, splits)
    with pytest.raises(ValueError, match="must be a Split"):
        validation.walk_forward(good, 10, [(np.arange(2), np.arange(2, 4))])


def test_walk_forward_never_selects_an_infinite_score():
    # A parameter that never traded has a flat curve and a +inf Sharpe; it
    # must not beat the finite scores of the ones that did.
    splits = validation.rolling(n=10, train=4, test=2)
    scores = {"idle": math.inf, "slow": 0.5, "fast": 1.5}
    result = validation.walk_forward(lambda rows: scores, 10, splits)
    assert [r["chosen"] for r in result] == ["fast"] * len(splits)


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: validation.rolling(10, 0, 2), "train must be >= 1"),
        (lambda: validation.rolling(10, 4, 0), "test must be >= 1"),
        (lambda: validation.rolling(10, 4, 2, step=0), "step must be >= 1"),
        (lambda: validation.rolling(5, 4, 2), "exceeds n"),
        (lambda: validation.rolling(10, 4.0, 2), "must be an integer"),
        (lambda: validation.rolling(10, True, 2), "must be an integer"),
        (lambda: validation.expanding(10, 0, 2), "min_train must be >= 1"),
        (lambda: validation.expanding(5, 4, 2), "exceeds n"),
        (lambda: validation.purged_kfold(10, 1), "k must be >= 2"),
        (lambda: validation.purged_kfold(3, 4), "exceeds n"),
        (lambda: validation.purged_kfold(10, 2, purge=-1), "purge must be >= 0"),
        (lambda: validation.purged_kfold(10, 2, embargo=-1), "embargo must be >= 0"),
        (lambda: validation.purged_kfold(10, 2, embargo=5), "no train rows"),
        (lambda: validation.rolling(-1, 1, 1), "n must be >= 0"),
    ],
)
def test_argument_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_numpy_integer_arguments_accepted():
    splits = validation.rolling(np.int64(10), np.int32(4), np.int64(3))
    assert len(splits) == 2
    assert "Split(train=[0:4], test=[4:7])" == repr(splits[0])
