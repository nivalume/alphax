# alphax

A parallel, factor-driven vectorized backtesting engine in Rust.

**Built for three things:**

1. Parallel parameter sweeps.
2. Parallel backtesting across independent symbols or multi-asset accounts.
3. Factor expressions, so factor libraries can be backtested at scale.

## Why

**A simulation loop is a branchy scalar loop.** Along the time axis there is a
hard loop-carried dependency, and across columns SIMD is a bad trade because
the execution kernel is a deep tree of data-dependent early exits. So the
speed does not come from vectorizing one run. It comes from a compiled kernel
with stack-allocated state, column-major arrays so a column is one contiguous
slice, and parallelism across columns, which are independent by construction.

**Keeping the engine thread-safe is a hard constraint.** Every engine instance
is `Send` with no global state, so a parameter sweep is threads in one process
rather than many processes, and a serial and a threaded run produce
bit-identical records. Random draws are functions of `(seed, row, col)`, so
they do not depend on thread interleaving.

So the opportunity is parallel sweeps and multi-asset accounts with a factor
pipeline feeding them.

## Performance

The simulation loop alone, on a 4-core Xeon (`benchmarks/`), in millions of
elements (bars × columns) per second, best of seven:

| Workload | 1 thread | 4 threads |
|---|---:|---:|
| 200000 × 1, every bar fills | 16.7M | 16.6M |
| 200000 × 1, one bar in 20 fills | 36.8M | 36.7M |
| 2000 × 500, every bar fills | 16.6M | 29.9M |
| 2000 × 500, one bar in 20 fills | 35.7M | 132.0M |
| 1000 × 2000, one bar in 20 fills | 35.0M | 133.3M |
| 20000 × 100, one bar in 20 fills | 35.6M | 131.9M |

A single column cannot use more than one thread, since columns are the
parallel axis. Sparse runs are compute-bound and scale almost linearly with
threads; dense runs, where every bar writes a record, are memory-bound and
scale less. The committed `benchmarks/baseline.tsv` is a regression gate, not a
cross-machine comparison: absolute numbers move with how busy the machine is.
See [benchmarks/README.md](benchmarks/README.md).

## Design

```
Factor expressions  →  factor panel (time × symbol)
                    →  signals (time × (symbol × parameter))
                    →  simulation   ← threaded across independent account groups
                    →  records      ← one schema per record kind
                    →  metrics per account, records per symbol
```

Parameters and symbols share a flat column axis. Columns are independent by
default; cash-sharing groups combine symbols into portfolios, with parameter
groups simulated in parallel. Python supports both orders and signals with
account-level results; see [shared accounts](crates/alphax-py/README.md#shared-accounts).

The analysis layer consumes records, not simulation state. Anything that
changes cash or a position is a record, and balances, value, returns and
metrics are rebuilt from records alone, so a result can be audited from the
records it came from.

For a guided path through the source itself (reading order per crate, the core algorithm of each
module, class and sequence diagrams, exercises and further reading), see the
[source knowledge base](https://alphax-docs.vercel.app) (中文), kept in [nivalume/alphax-docs](https://github.com/nivalume/alphax-docs). It is plain
Markdown that reads on GitHub; the site adds search and a graph of how the pages relate.

## What is in it

Bar/OHLCV data. The float tolerance layer, `FlexArray`, multi-level column
labels, the execution kernel, the `from_orders` and `from_signals` loops with
parallel execution across account groups, the stop engine, order records, all
three trade views, and the analysis layer — the value/returns rebuild chain,
return metrics, trade statistics and by-level aggregation — plus the array
layer (rolling, expanding and exponential windows), the eight indicators with
their parameter sweep, and the signal layer that produces entries and exits
rather than only consuming them. The factor DSL is implemented through
Alpha158 and Alpha360, and the Python package provides the external
market-data boundary.

### Factor evaluation

`FactorSet::evaluate` returns every factor as one `f64` matrix. Intermediates
are released after their last use, so on Alpha158 at 2520×500 nearly all of
the ~1.7 GB peak is the 158 output matrices themselves.
`FactorSet::evaluate_batches` evaluates a batch of symbols at a time and hands
each batch to a caller-supplied sink, so peak memory is bounded by the batch
rather than the panel.

The native evaluator compiles a factor set into one dependency graph and runs
it one symbol at a time, symbols in parallel. Cross-sectional operators
(`CSRank`, `CSZScore`) read every symbol on a row, so the graph is split at
those nodes into stages: symbols in parallel within a stage, one
cross-sectional pass over every symbol between stages. A set with a
cross-sectional operator needs the whole panel before it can hand anything
over in batches.

## Install

```bash
pip install alphax-trader            # extras: [data], [plot], [pandas]
```

The PyPI project is `alphax-trader`; the module is still `import alphax`.
Wheels cover Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows
(x86_64), CPython 3.9 and later.

## Try it

```bash
python -m pip install -e crates/alphax-py
python examples/sweep.py
```

Seven parameter pairs across four symbols: one run of twenty-eight columns,
threaded, with the result attributed back to the window and symbol that
produced it. The printed returns describe a synthetic generator, not a
strategy — the example says so, and says why that warning is there.

For shared cash across multiple symbols, run the offline
[portfolio example](examples/portfolio.py):

```bash
python examples/portfolio.py
```

It rebalances BTC/ETH/SOL in one account with `from_orders`, then compares two
moving-average parameter sets using `from_signals` and `group_lens=[3, 3]`.
Each parameter set gets an independent 100,000 account. The example prints
account equity, cash, positions and statistics, and checks that equity equals
cash plus the symbols' market value. Prices are synthetic; no downloads needed.

For Bars + funding + news from a catalog or standalone files:

```bash
python -m pip install -e 'crates/alphax-py[data]'
python examples/local_data.py                   # the data layer on its own, offline
python examples/multi_data.py --source catalog  # also parquet / csv
python examples/crypto.py --demo
# Full USD-M history from Binance Vision (writes ~/.cache/alphax/catalog):
python examples/download_binance_vision.py
```

The offline data example uses two complete `symbol.venue` identities and four
parameter pairs. Prices stay `[T,2]`; signals are `[T,8]`, with `column_map`
sharing prices across parameters. See the [data API](crates/alphax-py/README.md#local-data-and-parameter-sweeps).

## Developing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
pytest crates/alphax-py/tests -q
```

Local CSV/Parquet reading and alignment live in Python (`alphax.data`), on
pyarrow and NumPy. User Schemas name the columns and decode each Arrow batch
before simulation; the Rust compute crates stay independent of file I/O.
Binance HTTP and cache writing live only in examples. Parallel account
execution uses `std::thread::scope`.

To release, set `version` in `Cargo.toml` (`[workspace.package]`) and
`__version__` in `crates/alphax-py/python/alphax/__init__.py`, commit, and
push a matching tag (`git tag v0.1.0 && git push origin v0.1.0`).
[`release.yml`](.github/workflows/release.yml) builds the wheels and sdist,
publishes them to PyPI and creates the GitHub Release.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

The Alpha158 and Alpha360 factor expressions come from
[Qlib](https://github.com/microsoft/qlib), which is MIT-licensed, and the
fixtures that pin them were generated with it. The WorldQuant 101 alphas are
translated from the paper Kakushadze, "101 Formulaic Alphas" (arXiv:1601.00991).
Every other line here is an independent implementation.
