"""Chunked runs with checkpoints, column-distributed runs, and the cache."""

import numpy as np
import pytest

import alphax


def _market(n=300, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    close = np.asfortranarray(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, cols)), axis=0)))
    entries = np.asfortranarray(rng.random((n, cols)) < 0.1)
    exits = np.asfortranarray(rng.random((n, cols)) < 0.1)
    return close, entries, exits


def _same(a, b):
    oa, ob = a.orders(), b.orders()
    for k in ("idx", "col", "size", "price", "fees", "side"):
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)
    np.testing.assert_array_equal(a.value(), b.value())


@pytest.mark.parametrize("chunk", [1, 37, 100, 300])
def test_chunked_runs_are_the_one_run(chunk):
    close, entries, exits = _market()
    kw = dict(size=1.0, fees=0.001, sl_stop=0.05, direction="both", high=close * 1.01, low=close * 0.99)
    one = alphax.from_signals(close, entries, exits, **kw)
    _same(alphax.chunked.run("from_signals", close, entries, exits, chunk=chunk, **kw), one)
    size = np.where(entries, 1.0, np.where(exits, -1.0, np.nan))
    _same(alphax.chunked.run("from_orders", close, size, chunk=chunk), alphax.from_orders(close, size))


def test_chunked_runs_with_shared_cash():
    close, entries, exits = _market()
    kw = dict(size=1.0, cash_sharing=True, group_lens=[4, 2])
    one = alphax.from_signals(close, entries, exits, **kw)
    out = alphax.chunked.run("from_signals", close, entries, exits, chunk=37, **kw)
    assert out.cash_sharing and out.group_lens == (4, 2)
    _same(out, one)


def test_a_checkpoint_resumes_where_it_stopped(tmp_path, monkeypatch):
    close, entries, exits = _market()
    one = alphax.from_signals(close, entries, exits, size=1.0)
    ckpt = tmp_path / "run.ckpt"
    calls = {"n": 0}
    real = alphax.from_signals

    def failing(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(alphax, "from_signals", failing)
    with pytest.raises(KeyboardInterrupt):
        alphax.chunked.run("from_signals", close, entries, exits, chunk=50, checkpoint=ckpt, size=1.0)
    monkeypatch.setattr(alphax, "from_signals", real)
    resumed = alphax.chunked.run("from_signals", close, entries, exits, chunk=50, checkpoint=ckpt, size=1.0)
    _same(resumed, one)
    with pytest.raises(ValueError, match="another run"):
        alphax.chunked.run("from_signals", close, entries, exits, chunk=60, checkpoint=ckpt, size=1.0)


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_distributed_columns_are_the_one_run(backend):
    close, entries, exits = _market()
    one = alphax.from_signals(close, entries, exits, size=1.0, fees=np.full(6, 0.001))
    split = alphax.distributed.run("from_signals", close, entries, exits, n_chunks=3, backend=backend,
                                   size=1.0, fees=np.full(6, 0.001))
    _same(split, one)
    shared = dict(size=0.5, size_type="percent", cash_sharing=True, group_lens=[2, 2, 2])
    _same(alphax.distributed.run("from_signals", close, entries, exits, n_chunks=2, backend=backend, **shared),
          alphax.from_signals(close, entries, exits, **shared))


def test_chunks_keep_groups_whole():
    assert alphax.distributed.chunks([1] * 6, 3) == [(0, 2, 0, 2), (2, 4, 2, 4), (4, 6, 4, 6)]
    assert alphax.distributed.chunks([3, 1, 2], 2) == [(0, 1, 0, 3), (1, 3, 3, 6)]
    assert alphax.distributed.chunks([2], 5) == [(0, 1, 0, 2)]


def test_cache_counts_hits_and_survives_a_new_process(tmp_path):
    close, entries, exits = _market()
    cache = alphax.cache.Cache(tmp_path)
    calls = []

    @cache.memoize
    def run(threshold, e):
        calls.append(threshold)
        return alphax.from_signals(close, e, exits, size=threshold)

    a = run(1.0, entries)
    b = run(1.0, entries)
    run(2.0, entries)
    run(1.0, ~entries)
    assert calls == [1.0, 2.0, 1.0] and b is a
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 3
    fresh = alphax.cache.Cache(tmp_path)
    again = fresh.memoize(run.__wrapped__)(1.0, entries)
    assert fresh.stats()["disk_hits"] == 1
    _same(again, a)
    fresh.clear()
    assert not list(tmp_path.glob("*.pkl"))
    assert alphax.cache.digest(np.arange(3)) != alphax.cache.digest(np.arange(3.0))


def test_distributed_on_ray_when_installed():
    ray = pytest.importorskip("ray")
    close, entries, exits = _market()
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        _same(alphax.distributed.run("from_signals", close, entries, exits, n_chunks=2, backend="ray", size=1.0),
              alphax.from_signals(close, entries, exits, size=1.0))
    finally:
        ray.shutdown()


def test_a_checkpoint_from_other_data_is_refused(tmp_path):
    close, entries, exits = _market()
    ckpt = tmp_path / "run.ckpt"
    alphax.chunked.run("from_signals", close, entries, exits, chunk=100, checkpoint=ckpt, size=1.0)
    with pytest.raises(ValueError, match="another run"):
        alphax.chunked.run("from_signals", close * 1.01, entries, exits, chunk=100, checkpoint=ckpt, size=1.0)


def test_shared_deposits_follow_their_groups():
    close, entries, exits = _market()
    deposits = np.zeros((close.shape[0], 3))
    deposits[50] = [100.0, 200.0, 300.0]
    kw = dict(size=1.0, cash_sharing=True, group_lens=[2, 2, 2], cash_deposits=deposits)
    _same(alphax.distributed.run("from_signals", close, entries, exits, n_chunks=3, backend="serial", **kw),
          alphax.from_signals(close, entries, exits, **kw))


def test_the_cache_sees_an_edited_function(tmp_path):
    def f(x):
        return x + 1

    key = alphax.cache.digest(f, (1,), {})

    def f(x):  # noqa: F811 - the edit
        return x + 2

    assert alphax.cache.digest(f, (1,), {}) != key


def test_a_checkpoint_extends_a_finished_run_over_new_bars(tmp_path):
    close, entries, exits = _market()
    ckpt = tmp_path / "run.ckpt"
    first = alphax.chunked.run("from_signals", close[:120], entries[:120], exits[:120], chunk=50,
                               checkpoint=ckpt, size=1.0)
    _same(first, alphax.from_signals(close[:120], entries[:120], exits[:120], size=1.0))
    longer = alphax.chunked.run("from_signals", close, entries, exits, chunk=50, checkpoint=ckpt, size=1.0)
    _same(longer, alphax.from_signals(close, entries, exits, size=1.0))
    edited = close.copy()
    edited[10] *= 1.05
    with pytest.raises(ValueError, match="another run"):
        alphax.chunked.run("from_signals", edited, entries, exits, chunk=50, checkpoint=ckpt, size=1.0)
