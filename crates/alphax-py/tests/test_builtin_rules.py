"""The built-in from_signals rules against the same rules as Python callbacks."""

import numpy as np
import pytest

import alphax


def _market(n=250, cols=5, seed=3):
    rng = np.random.default_rng(seed)
    close = np.asfortranarray(100 * np.exp(np.cumsum(rng.normal(0, 0.015, (n, cols)), axis=0)))
    high, low = close * (1 + rng.uniform(0, 0.01, close.shape)), close * (1 - rng.uniform(0, 0.01, close.shape))
    entries = np.asfortranarray(rng.random((n, cols)) < 0.08)
    exits = np.asfortranarray(rng.random((n, cols)) < 0.04)
    return close, np.asfortranarray(high), np.asfortranarray(low), entries, exits


def _same(a, b):
    oa, ob = a.orders(), b.orders()
    for k in ("idx", "col", "size", "price", "side"):
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)


def test_risk_sizing_matches_the_callback_it_replaces():
    close, high, low, entries, exits = _market()
    kw = dict(high=high, low=low, init_cash=10_000, sl_stop=0.05)
    built = alphax.from_signals(close, entries, exits, risk=0.01, risk_stop=0.05, **kw)

    def signal(ctx, k, state):
        if ctx.position[k] == 0:
            price = ctx.close[k]
            # alphax.risk_size's arithmetic: the distance to the stop price.
            return {"size": 0.01 * ctx.value / abs(price - price * (1 - 0.05)), "size_type": "amount"}
        return None

    py = alphax.from_signals(close, entries, exits, signal_func=signal, **kw)
    _same(built, py)
    o = built.orders()
    first = o["idx"] == o["idx"].min()
    assert np.allclose(o["size"][first] * o["price"][first] * 0.05, 100.0)


def test_breakeven_moves_the_stop_to_the_entry():
    close, high, low, entries, exits = _market()
    kw = dict(high=high, low=low, init_cash=1e6, size=1.0, sl_stop=0.10)
    plain = alphax.from_signals(close, entries, exits, **kw)
    moved = alphax.from_signals(close, entries, exits, breakeven=0.02, **kw)

    def adjust(ctx, k, stops, state):
        pos, entry = ctx.position[k], stops["tp_price"]
        if pos == 0 or not np.isfinite(entry):
            return None
        last = ctx.val_price[k]
        gain = last / entry - 1 if pos > 0 else 1 - last / entry
        level = stops["sl_price"] * (1 - stops["sl_stop"]) if pos > 0 else stops["sl_price"] * (1 + stops["sl_stop"])
        if gain >= 0.02 and not (level >= entry if pos > 0 else level <= entry):
            return {"sl_price": entry, "sl_stop": 0.0, "sl_trail": False}
        return None

    _same(moved, alphax.from_signals(close, entries, exits, adjust_func=adjust, **kw))
    t, p = moved.trades(), plain.trades()
    assert len(t["entry_idx"]) >= len(p["entry_idx"])
    # Some trade now exits at its entry price.
    assert np.isclose(t["exit_price"], t["entry_price"]).any()


def test_budget_caps_open_positions_and_exposure():
    close, high, low, entries, exits = _market(cols=8)
    shared = dict(high=high, low=low, size=0.1, size_type="value", init_cash=1.0, cash_sharing=True,
                  group_lens=[8], call_seq="auto")
    pf = alphax.from_signals(close, entries, exits, max_positions=2, **dict(shared, init_cash=10.0))
    o = pf.orders()
    pos = np.zeros(8)
    held = []
    for i in range(close.shape[0]):
        for k in np.flatnonzero(o["idx"] == i):
            pos[o["col"][k]] += o["size"][k] * (1 if o["side"][k] == 0 else -1)
        held.append((np.abs(pos) > 1e-12).sum())
    assert max(held) == 2
    free = alphax.from_signals(close, entries, exits, **dict(shared, init_cash=10.0))
    assert len(free.orders()["idx"]) > len(o["idx"])
    capped = alphax.from_signals(close, entries, exits, max_exposure=0.15, **dict(shared, init_cash=1.0))
    assert len(capped.orders()["idx"]) < len(alphax.from_signals(close, entries, exits, **shared).orders()["idx"])


def test_rules_are_checked_and_thread_invariant():
    close, high, low, entries, exits = _market()
    with pytest.raises(ValueError, match="both risk and risk_stop"):
        alphax.from_signals(close, entries, exits, risk=0.01)
    with pytest.raises(ValueError, match="cannot be combined"):
        alphax.from_signals(close, entries, exits, breakeven=0.01, signal_func=lambda c, k, s: None)
    kw = dict(high=high, low=low, sl_stop=0.05, risk=0.02, risk_stop=0.05, breakeven=0.01, init_cash=1e4)
    _same(alphax.from_signals(close, entries, exits, threads=1, **kw),
          alphax.from_signals(close, entries, exits, threads=3, **kw))
