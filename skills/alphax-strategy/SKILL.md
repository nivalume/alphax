---
name: alphax-strategy
description: Write, run and validate a trading strategy with alphax's Python API — signals from factor expressions or NumPy, from_signals or from_orders, parameter sweeps, walk-forward and significance checks. Use when asked to backtest an idea, write a strategy, sweep its parameters, or check whether a result is real, in this repository or with the installed alphax package.
license: Apache-2.0 (the repository licence)
compatibility: Python 3.9+ with the alphax package (uv sync in this repository).
metadata:
  author: alphax maintainers
  version: "1.0"
  canonical-path: skills/alphax-strategy
---

# Writing a strategy with alphax

Look up any function before using it: `alphax.help("from_signals")`,
`alphax.docs.search("trailing stop")`, or the MCP server's `api_reference`.
Arrays are `(bars, columns)`; one column is one account unless `cash_sharing`
groups them.

## 1. Signals

Prefer factor expressions: they are compiled once, run in Rust, and say what
they mean.

```python
import numpy as np, alphax

fields = {"close": close, "high": high, "low": low, "volume": volume}   # (bars, symbols), float
f = alphax.FactorSet.compile({
    "fast": "Mean($close, 10)", "slow": "Mean($close, 50)",
    "rank": "CSRank(Delta($close, 20))",              # cross-sectional: across symbols, per bar
}).evaluate(fields)
entries, exits = f["fast"] > f["slow"], f["fast"] < f["slow"]
```

`Mean` and the other rolling operators follow Qlib: they report from the first
bar on a partial window. Mask the warm-up yourself if that matters
(`entries[:50] = False`). Built-in sets: `alphax.alpha158()`, `alpha360()`,
`wq101()`.

## 2. Run

```python
pf = alphax.from_signals(close, entries, exits, high=high, low=low,
                         size=0.5, size_type="percent", fees=0.0005,
                         sl_stop=0.05, init_cash=10_000)
pf.stats(0); pf.total_return(); pf.orders(); pf.trades()
```

- Stops need `high` and `low` (and `open` for gaps), or the run refuses them.
- Signals act at the bar's close unless `price=` says otherwise. If the signal
  reads the bar's close, delay it: `@alphax.strategy` does that with
  `signal_delay=1` by default.
- Weights instead of signals: `alphax.allocation.weights(close, fn, every=21)`
  then `allocation.from_weights(close, w)`.
- `alphax.check_shapes(close, size=..., fees=...)` tells how each input will
  broadcast before you run.

## 3. Sweep

```python
from alphax import params
combos = params.grid(fast=[5, 10, 20], slow=[50, 100], where=lambda p: p["fast"] < p["slow"])
res = params.run(lambda fast, slow: run(fast, slow).sharpe_ratio().mean(), combos, n_jobs=4)
res.best(); params.neighbors(res)        # a peak with weak neighbours is noise
```

For many columns at once, one run over all parameter columns is faster than a
loop: `@alphax.strategy` + `.sweep(data, params=combos)`.

## 4. Check that it is real

Do these before reporting a number; report the ones you ran.

| Question | Call |
|---|---|
| Out of sample? | `alphax.validation.walk_forward(fn, n, validation.rolling(n, train, test))` or `validation.calendar(index, "M", train=12)` |
| Luck? | `validation.permutation_test(positions, returns)`, `validation.shuffled_test(fn, close)` |
| Selection bias over many trials? | `alphax.robust.deflated_sharpe(returns, trial_sharpes)`, `robust.pbo_cscv(perf)` |
| Reading its own bar? | `validation.signal_delay_test(strategy, data, params, delays=(0, 1, 2))`, or with positions only `validation.position_delay_test(positions, returns)` |
| Does the signal predict, and for how long? | `predictions.power(signal, forward_returns)` (R², quantile returns), `predictions.decay(signal, close)` |
| Is the drawdown just the luck of the order? | `robust.bootstrap(returns, block=...)`; `returns.stats(r)["max_drawdown_recovery"]` |
| Made on a few bars? | `returns.metric(r, "return_concentration")`, `returns.metric(r, "hit_ratio")` |
| New to the book? | `allocation.marginal_contribution(book, candidate)` |

A Sharpe that collapses from delay 0 to delay 1 is a look-ahead, not an edge.

## 5. Report

Give the parameters, the period, costs assumed, the in-sample and
out-of-sample figures separately, and the checks from §4 with their p-values.
Save the run: `alphax.save(pf, "run.alphax")`; `alphax.load` gives it back.
