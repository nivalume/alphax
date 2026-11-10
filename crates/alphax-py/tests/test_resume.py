"""Continuing a portfolio (resume=pf), and saving and loading one."""

import numpy as np
import pytest

import alphax


def _market(n=80, cols=3, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, cols)), axis=0))
    entries = rng.random((n, cols)) < 0.15
    exits = rng.random((n, cols)) < 0.15
    size = np.where(rng.random((n, cols)) < 0.3, rng.normal(0, 1, (n, cols)), np.nan)
    return np.asfortranarray(close), np.asfortranarray(entries), np.asfortranarray(exits), size


def _same(a, b):
    oa, ob = a.orders(), b.orders()
    assert oa.keys() == ob.keys()
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)
    np.testing.assert_array_equal(a.value(), b.value())
    np.testing.assert_array_equal(a.cash(), b.cash())


@pytest.mark.parametrize("cut", [1, 30, 79])
def test_from_signals_resumed_is_the_one_run(cut):
    close, entries, exits, _ = _market()
    kw = dict(
        size=1.0,
        sl_stop=0.03,
        tp_stop=0.05,
        fees=0.001,
        cash_earnings=0.1,
        group_lens=[2, 1],
        cash_sharing=True,
        log=True,
    )
    hl = dict(high=close * 1.01, low=close * 0.99)
    one = alphax.from_signals(close, entries, exits, **hl, **kw)
    head = alphax.from_signals(
        close[:cut], entries[:cut], exits[:cut], high=hl["high"][:cut], low=hl["low"][:cut], **kw
    )
    assert head.last_state.next_row == cut
    whole = alphax.from_signals(close, entries, exits, **hl, resume=head, **kw)
    _same(whole, one)
    la, lb = whole.logs(), one.logs()
    for k in la:
        np.testing.assert_array_equal(la[k], lb[k], err_msg=k)
    assert whole.last_state.to_bytes() == one.last_state.to_bytes()


def test_from_orders_and_from_order_func_resumed():
    close, _, _, size = _market(seed=3)
    kw = dict(fees=0.002, leverage=2.0, init_cash=1_000.0, reject_prob=0.1, seed=4)
    one = alphax.from_orders(close, size, **kw)
    head = alphax.from_orders(close[:41], size[:41], **kw)
    _same(alphax.from_orders(close, size, resume=head, **kw), one)

    def order_func(ctx, k, state):
        return size[ctx.row, ctx.from_col + k]

    one = alphax.from_order_func(close, order_func, **kw)
    head = alphax.from_order_func(close[:41], order_func, **kw)
    _same(alphax.from_order_func(close, order_func, resume=head, **kw), one)


def test_save_and_load(tmp_path):
    close, entries, exits, _ = _market(seed=5)
    kw = dict(size=2.0, fees=0.001, log=True)
    head = alphax.from_signals(close[:50], entries[:50], exits[:50], **kw)
    path = tmp_path / "pf.alphax"
    alphax.save(head, path)
    loaded = alphax.load(path)
    _same(loaded, head)
    assert repr(loaded.stats()) == repr(head.stats())
    # A loaded portfolio continues as the one it was saved from.
    one = alphax.from_signals(close, entries, exits, **kw)
    _same(alphax.from_signals(close, entries, exits, resume=loaded, **kw), one)
    with pytest.raises(ValueError, match="not alphax portfolio"):
        alphax.Portfolio.from_bytes(b"something else")
    data = head.to_bytes()
    with pytest.raises(ValueError, match="cut short"):
        alphax.Portfolio.from_bytes(data[:-3])


def test_state_bytes():
    close, entries, exits, _ = _market()
    pf = alphax.from_signals(close, entries, exits, size=1.0)
    state = pf.last_state
    back = alphax.SimState.from_bytes(state.to_bytes())
    assert back.to_bytes() == state.to_bytes()
    assert back.next_row == len(close)
    assert len(back.position) == close.shape[1]
    with pytest.raises(ValueError, match="version"):
        bad = bytearray(state.to_bytes())
        bad[4] = 2
        alphax.SimState.from_bytes(bytes(bad))


def test_what_resume_refuses():
    close, entries, exits, _ = _market()
    head = alphax.from_signals(close[:30], entries[:30], exits[:30], size=1.0)
    other = close.copy(order="F")
    other[10, 1] *= 1.01
    with pytest.raises(ValueError, match="bar 10 of column 1"):
        alphax.from_signals(other, entries, exits, size=1.0, resume=head)
    with pytest.raises(ValueError, match="whole history"):
        alphax.from_signals(close[:20], entries[:20], exits[:20], size=1.0, resume=head)
    with pytest.raises(ValueError, match="grouping"):
        alphax.from_signals(
            close, entries, exits, size=1.0, group_lens=[3], cash_sharing=True, resume=head
        )
    with pytest.raises(ValueError, match="cannot continue"):
        alphax.from_orders(close, 1.0, resume=head)
    with pytest.raises(TypeError, match="Portfolio or a SimState"):
        alphax.from_signals(close, entries, exits, resume="pf")
    with pytest.raises(ValueError, match="whole history, which starts at row 0"):
        alphax.from_signals(close, entries, exits, resume=head, row_offset=3)
    with pytest.raises(ValueError, match="pass the state as resume"):
        alphax.from_signals(close, entries, exits, row_offset=3)
    with pytest.raises(ValueError, match="auto"):
        alphax.from_signals(close, entries, exits, init_cash="auto", resume=head)


def test_a_state_continues_on_arrays_from_later():
    close, entries, exits, _ = _market()
    one = alphax.from_signals(close, entries, exits, size=1.0, price=-np.inf)
    head = alphax.from_signals(close[:40], entries[:40], exits[:40], size=1.0, price=-np.inf)
    # The rows from 39: the previous close the first new bar's price reads.
    seg = alphax.from_signals(
        close[39:], entries[39:], exits[39:], size=1.0, price=-np.inf,
        resume=head.last_state, row_offset=39,
    )
    assert isinstance(seg, alphax.Segment)
    assert seg.last_state.to_bytes() == one.last_state.to_bytes()
    o, s = one.orders(), seg.orders()
    later = o["idx"] >= 40
    for k in o:
        np.testing.assert_array_equal(o[k][later], s[k], err_msg=k)
