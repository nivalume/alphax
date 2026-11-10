"""Shared accounts: public API and the committed golden outputs."""

import json
from pathlib import Path

import alphax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/shared_accounts.json"


def make_portfolio(case, threads=1):
    args = dict(case["inputs"])
    return getattr(alphax, f"from_{case['kind']}")(
        **args, cash_sharing=True, threads=threads
    )


def same(actual, expected, label):
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True, err_msg=label
    )


def check(pf, expected):
    for name in ("orders", "logs", "trades"):
        actual = getattr(pf, name)()
        for key, values in expected[name].items():
            field = np.asarray(actual["ret" if key == "return" else key])
            if field.dtype.kind in "biu":
                np.testing.assert_array_equal(field, values, err_msg=f"{name}.{key}")
            else:
                same(field, values, f"{name}.{key}")
    for name in ("cash", "value", "returns"):
        actual = np.column_stack(
            [getattr(pf, name)(group=g) for g in range(pf.n_groups)]
        )
        same(actual, expected[name], name)
    same(
        np.column_stack([pf.assets(c) for c in range(pf.n_cols)]),
        expected["assets"],
        "assets",
    )
    actual = pf.metrics()
    for name, values in expected["metrics"].items():
        same(actual[name], values, name)
    same(actual["trade_count"], expected["trade_count"], "trade_count")
    for group, expected_stats in enumerate(expected["stats"]):
        actual = pf.stats(group=group)
        assert actual["Benchmark Return [%]"] is None
        for name, value in expected_stats.items():
            if name != "Benchmark Return [%]":
                same(actual[name], value, f"stats[{group}].{name}")


CASES = json.loads(FIXTURE.read_text())["cases"] if FIXTURE.exists() else []


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_committed_shared_account_golden(case):
    pf = make_portfolio(case)
    assert pf.cash_sharing
    assert pf.group_lens == tuple(case["inputs"]["group_lens"])
    assert pf.n_groups == len(pf.group_lens)
    check(pf, case["expected"])
    parallel = make_portfolio(case, threads=3)
    check(parallel, case["expected"])


def test_fixture_has_both_paths_and_edge_cases():
    assert len(CASES) >= 40
    assert {c["kind"] for c in CASES} == {"signals", "orders"}


def test_public_account_boundary_and_owned_results():
    close = np.full((3, 2), 10.0)
    pf = alphax.from_orders(close, size=2.0, init_cash=100.0, cash_sharing=True)
    assert pf.n_cols == 2 and pf.n_groups == 1
    assert pf.group_lens == (2,)
    same(pf.value(group=0), [100.0, 100.0, 100.0], "capital once")
    with pytest.raises(ValueError, match="group="):
        pf.value(0)
    with pytest.raises(ValueError, match="group="):
        pf.stats(col=0)
    with pytest.raises(ValueError, match="not both"):
        pf.returns(col=0, group=0)
    with pytest.raises(IndexError):
        pf.cash(group=1)
    with pytest.raises(ValueError):
        pf.traded_only(np.ones(2))
    for key, value in pf.stats().items():
        if value is None:
            assert pf.stats(group=0)[key] is None
        else:
            same(value, pf.stats(group=0)[key], key)
    value = pf.value(group=0)
    value[:] = -1
    same(pf.value(group=0), [100.0, 100.0, 100.0], "owned results")
    with pytest.raises(AttributeError):
        pf.group_lens = (1, 1)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"group_lens": [2]}, ValueError),
        ({"cash_sharing": True, "group_lens": [1]}, ValueError),
        (
            {"cash_sharing": True, "group_lens": [1, int(np.iinfo(np.uintp).max)]},
            ValueError,
        ),
        ({"cash_sharing": True, "group_lens": [0, 2]}, ValueError),
        ({"cash_sharing": True, "group_lens": [-1, 3]}, ValueError),
        ({"cash_sharing": True, "group_lens": []}, ValueError),
        ({"cash_sharing": True, "group_lens": [True, 1]}, TypeError),
        ({"cash_sharing": True, "group_lens": [2.0]}, TypeError),
        ({"cash_sharing": True, "init_cash": [100.0, 200.0]}, ValueError),
        ({"cash_sharing": True, "init_cash": [[100.0]]}, ValueError),
        ({"cash_sharing": True, "call_seq": "random"}, ValueError),
    ],
)
@pytest.mark.parametrize("kind", ["signals", "orders"])
def test_invalid_account_inputs(kind, kwargs, error):
    close = np.full((3, 2), 10.0)
    args = dict(close=close, **kwargs)
    if kind == "signals":
        args.update(
            entries=np.ones_like(close, dtype=bool),
            exits=np.zeros_like(close, dtype=bool),
        )
    with pytest.raises(error):
        getattr(alphax, f"from_{kind}")(**args)


def test_percent_is_remaining_cash_not_equal_weights():
    close = np.full((1, 2), 10.0)
    pf = alphax.from_signals(
        close,
        [[True, True]],
        [[False, False]],
        size=0.5,
        size_type="percent",
        cash_sharing=True,
        init_cash=100.0,
    )
    same(pf.orders()["size"], [5.0, 2.5], "sequential percentages")
    same(pf.cash(group=0), [25.0], "remaining cash")
