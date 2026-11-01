# phasor-py

Python bindings. PyO3 + rust-numpy, zero-copy in, NumPy arrays out.

```python
import numpy as np, phasor

pf = phasor.from_signals(close, entries, exits, init_cash=10_000, fees=0.001, threads=8)
print(pf.stats())                     # the 28-row table, for one column
best = int(np.nanargmax(pf.traded_only(pf.sharpe_ratio())))

# One order per element instead of signals: a rebalance is a weight matrix.
pf = phasor.from_orders(close, weights, size_type="targetpercent")

# And when a run traded less than expected, ask why.
pf = phasor.from_signals(close, entries, exits, init_cash=10_000, log=True)
print(pf.reason_counts(0))            # {'filled': 31, 'no cash to go long': 12}
```

## Shared accounts

Both `from_orders` and `from_signals` accept `cash_sharing=False`,
`group_lens=None`, and `call_seq="default"`. With sharing enabled and no
lengths, all columns form one account. Explicit lengths partition adjacent
columns; `[3, 3]` represents two parameter sets over three symbols each.
`init_cash` is a scalar or one balance per group, counted once per account.
Without sharing, existing per-column semantics are unchanged and explicit
`group_lens` is rejected.

```python
# weights has the same (time, symbol) shape as close.
# Use NaN to skip a rebalance and zero to close a position.
pf = phasor.from_orders(
    close, weights, size_type="targetpercent",
    cash_sharing=True, init_cash=100_000, call_seq="auto", fees=0.001,
)
print(pf.metrics())             # one result per account
print(pf.stats(group=0))
print(pf.cash(group=0))
print(pf.value(group=0))
print(pf.assets(col=0))         # symbol-level quantity
```

`call_seq="auto"` stably sorts approximate order values so sells release cash
before buys. Equal values retain column order. It does not allocate equal
weights: `size_type="percent"` consumes a fraction of the resources remaining
at each fill; `targetpercent` targets a fraction of the account's valuation.
Targets are supported by `from_orders`; signals accept amount, value and
percent. Both paths accept `val_price`, `ffill_val_price` and `update_value`;
with the default `update_value=False`, targets within a bar use the same
pre-fill account valuation. Auto sequencing assumes the Bar model's prices
remain available regardless of execution order.

`n_cols` counts logical simulation columns; `n_groups`, `group_lens` and `cash_sharing`
report account metadata. Metrics and trade counts are per group. Shared
account series require `group=`; `col=` continues to select positions and
record diagnostics. `stats()` defaults to account 0. Ungrouped positional
calls such as `value(0)` remain supported. Orders, trades and logs retain
original symbol columns; log `group` identifies the cash account. Shared
accounts do not infer a benchmark, so `Benchmark Return [%]` is `None`.
An unheld symbol's missing price contributes zero value to a shared account;
a missing held price propagates NaN in closing valuation.

Groups are independent and can run on separate threads. Symbols within one
account execute sequentially. `python examples/portfolio.py` demonstrates
allocation and signal parameter sweeps without network access. Signal delays
and execution prices remain explicit; no risk or automatic allocation layer
is added.

## Deposits and earnings

Both loops accept `cash_deposits`, `cash_dividends` and `cash_earnings`: cash
that moves without a fill.

- `cash_deposits` is booked at each bar's open, before orders are sized, so a
  `targetpercent` rebalance on the same bar invests it. Positive adds, negative
  withdraws. With `cash_sharing` a matrix is `(n_bars, n_groups)`, one column per
  account.
- `cash_earnings` is booked at each bar's close, per column, including bars with
  no signal: interest, fees, or a funding payment computed in advance.
- `cash_dividends` is per share and shaped like `close` (`column_map` applies).
  It is paid at each bar's open on the position held before that bar's orders,
  so buying on the ex-date does not earn it; a short position pays it.
- A 1-D array is one value per bar (a schedule), not one value per column.
- A withdrawal or a charge larger than the free cash is applied in part.
  `pf.cash_records()` returns what was applied (`col`, `idx`, `amount`, and
  `kind`: 0 deposit, 1 earning, 2 dividend).
- Deposits are not returns: each bar's return is measured from the previous
  value plus that bar's deposit, total return compounds the bars, and
  drawdowns ignore money moved in or out.

```python
# 1,000 on the first bar of every month, invested by the monthly rebalance.
deposits = np.where(month_start, 1_000.0, 0.0)
pf = phasor.from_orders(close, weights, size_type="targetpercent",
                      cash_sharing=True, cash_deposits=deposits)
```

## Order controls

`cash_limit` caps the cash one buy may spend, fees included; it applies before
`percent` takes its share and does not affect sells or shorts. `reject_prob`
rejects orders at random with that probability (status `RandomEvent` in the
logs), for testing how a strategy degrades when fills fail. Each draw depends
only on `seed`, the bar and the column, so a run is reproducible on any thread
count and under any grouping.

## Leverage

`leverage` (1 or more, per column) lets a buy be worth that many times its own
cash. `leverage_mode="lazy"`, the default, spends own cash first and borrows
only what it cannot cover; `"eager"` borrows `1 - 1/leverage` of every buy.
Fees come out of own cash, `percent` sizes against the levered buying power,
and a short ties up its proceeds plus `value / leverage` instead of twice its
value. A buy with no own cash left is refused (`NoCashLeverage` in the logs).

- `pf.value()` is the equity: positions at market, less what the longs owe.
- `pf.cash()` is own cash less that borrowing, so it goes negative on a
  levered long; `pf.free_cash()` also takes out what shorts tie up, and
  `pf.gross_exposure()` rises above 1 with the leverage.
- `pf.orders()` gains `debt` and `locked_cash` per fill, and the logs gain
  `locked_cash`, `new_locked_cash`, `req_leverage` and `req_leverage_mode`.
- There is no interest on the borrowing and no liquidation yet.

```python
# Twice the cash in the first bar's buy: 1,000 of own money, 1,000 borrowed.
pf = phasor.from_orders(close, size=np.inf, init_cash=1_000.0, leverage=2.0)
```

## Margin accounts

`account_type="margin"` makes a column trade the way futures, perpetuals and
CFDs do: either side only posts initial margin out of free cash, and the
position's profit or loss is settled into cash at every bar. Give one type for
every column, or a list with one per column, so a spot column and a perpetual
column in one group share their cash. `initial_margin` is a fraction of the
notional (`margin_basis="notional"`, the default; left unset it is
`1 / leverage`) or cash per unit of size (`margin_basis="unit"`, as futures
exchanges publish it per contract). Closing is always allowed; opening needs
free cash for its margin and fees, and is refused (`NoCashLong` /
`NoCashShort`) once a loss has used it up.

- `pf.value()` is the equity; `pf.free_cash()` is that equity less the margin
  posted, and `pf.gross_exposure()` counts the whole contract over it.
- `pf.orders()` gains `debt` (the cost basis), `locked_cash` (the margin) and
  `margin` per fill, and the logs gain `req_account_type`,
  `req_initial_margin` and `req_margin_basis`.
- `maintenance_margin` liquidates: see below. `margin_mode="isolated"` keeps a
  position on its own margin, not settled into the account's cash.

```python
# Spot long, perpetual short on 10% margin, one account: at 120 one gains
# what the other loses, so the value stays 1,000.
close = np.array([[100.0, 100.0], [120.0, 120.0]])
size = np.array([[5.0, -5.0], [np.nan, np.nan]])
pf = phasor.from_orders(close, size=size, init_cash=1_000.0, cash_sharing=True,
                        account_type=["cash", "margin"],
                        initial_margin=np.array([np.nan, 0.1]))
```

## Contract multiplier

`multiplier` is what one unit of position is worth per unit of price: 50 for an
E-mini S&P 500 future, 1 (the default) for a share. A number, or one value per
column (per source column with `column_map`); it cannot change within a
column. A fill moves `size * price * multiplier` of cash, and fees, margin,
dividends, holding charges and the liquidation test all follow that notional.
Sizes, `min_size`, `size_granularity` and a `"unit"` margin stay in
contracts; prices, stops and `pf.orders()["price"]` stay prices. `pf.trades()`
carries `multiplier`, with `pnl` multiplied and `ret` not.

```python
# One E-mini on 12,000 of margin per contract: a 10-point move is 500.
pf = phasor.from_orders(close, size=size, init_cash=20_000.0, account_type="margin",
                        initial_margin=12_000.0, margin_basis="unit", multiplier=50.0)
```

## Liquidation

`maintenance_margin` (in `margin_basis`, like `initial_margin`) makes an
account liquidate when its equity falls below the maintenance margin of its
margin positions. Every bar, before its orders, each held column moves from the
bar's `open` to its adverse extreme (`low` for a long, `high` for a short), or to
`mark_price` when one is given, and the account is closed where it first
breaches, or at the open when the bar gaps through. The fill pays
`liquidation_fee` of its notional and is flagged in `pf.orders()["liquidation"]`
and in the logs. Without `high` and `low` the test uses the close.

- `margin_mode="cross"` (the default): the account's equity carries every cross
  position, cash columns count as collateral and are never closed, and the
  cross positions close together. A hedge tested at each leg's own extreme
  is liquidated more easily than at any one instant, so give it a mark price.
- `margin_mode="isolated"`: the position is not settled into the account's
  cash, is tested on its own margin and can lose no more than it posted.
- A loss the account cannot pay is written off, as an exchange's insurance fund
  takes it: `pf.cash_flows("liquidation")`.
- In `from_signals` a stop that fires on the bar before the liquidation price
  closes the position first; a liquidated column's pending signal is dropped.

```python
# A 10x perpetual long on OHLC bars, liquidated at a twentieth of the notional.
pf = phasor.from_orders(close, size=size, open=open_, high=high, low=low,
                        account_type="margin", leverage=10.0,
                        maintenance_margin=0.05, liquidation_fee=0.005)
pf.orders()["liquidation"].sum()
```

## Holding charges

What holding a position costs or earns, charged at each bar's close on the
position the bar's orders left, quiet bars included:

| Argument | Charged on | Amount | `pf.cash_flows(...)` |
|---|---|---|---|
| `funding_rate` | any position | `-position * price * rate`: longs pay a positive rate, shorts receive it | `"funding"` |
| `long_financing_rate`, `short_financing_rate` | any position | `-abs(position) * price * rate`, the rate of its side | `"financing"` |
| `debt_rate` | a cash column's leveraged long | `-debt * rate` | `"interest"` |
| `borrow_rate` | a cash column's short | `-abs(position) * price * rate` | `"borrowfee"` |

`price` is `mark_price`, or the close when it is not given. Every rate but
`debt_rate` is shaped like `close`, through `column_map`, as an instrument's own
data; `debt_rate` is a number, one value per bar or a matrix. Rates are per bar,
for the time until the next bar: `phasor.rates.per_bar(annual, index)` converts
an annual rate from the bars' timestamps, so Friday's daily bar charges three
days. A funding rate needs no conversion: put it on the bar whose close is its
funding time, `NaN` elsewhere.

A charge is owed whatever the balance, so unlike a `cash_earnings` charge it is
not capped: these runs keep balances like a leveraged one, cash can go below
zero, and then only closing is allowed. Margin columns pay funding and
financing, not interest or borrow fees. `pf.cash_records()` lists every charge;
`pf.cash_flows(kind)` gives one kind per bar and column, and all of them are
part of the return.

```python
# A spot long and a perpetual short in one account, the short collecting
# the funding the exchange publishes every eight hours.
pf = phasor.from_orders(close, size=size, init_cash=10_000.0, cash_sharing=True,
                        account_type=["cash", "margin"], leverage=np.array([1.0, 5.0]),
                        funding_rate=funding)     # (n_bars, 2), NaN off the hour
pf.cash_flows("funding")[:, 1].sum()              # what the hedge earned
```

## Factor expressions

### Function strategies

Use a Python function to turn a `BarPanel` or field mapping into entry and exit
signals. A direct call returns raw, undelayed signals. `run` and `sweep` delay
them one bar by default, then call the existing `from_signals` engine. With its
default execution price, a decision on bar `t` fills at bar `t+1`'s close.

```python
@phasor.strategy
def ma_cross(data, fast=10, slow=50):
    f = phasor.factors(data, fast=f"Mean($close, {fast})",
                     slow=f"Mean($close, {slow})")
    return (phasor.crossed_above(f["fast"], f["slow"]),
            phasor.crossed_above(f["slow"], f["fast"]))

entries, exits = ma_cross({"close": close}, fast=10, slow=50)
pf = ma_cross.run({"close": close}, params={"fast": 10, "slow": 50}, fees=0.001)
pf = ma_cross.sweep(
    {"close": close},
    params=[{"fast": 10, "slow": 50}, {"fast": 20, "slow": 100}],
    fees=0.001,
)
```

The strategy returns two Boolean arrays of shape `(time, symbols)` (a 1-D
array also works for one symbol). Each parameter set runs over every symbol;
result columns follow parameter order, then source symbol order. `signal_delay`
accepts a nonnegative bar count, including zero. `phasor.factors` uses the existing
factor DSL and rejects expressions marked as using future data. It does not
inspect arbitrary Python strategy code for lookahead.

For one account per parameter set, pass `cash_sharing=True`; the wrapper groups
each set's symbols automatically, leaving sets independent. Execution options
such as `size`, `fees`, `slippage`, `init_cash`, and `threads` pass through to
`from_signals`. OHLC fields come from the input panel or mapping. The wrapper
owns the signal arrays, price column map, and group boundaries.

`phasor.crossed_above` uses the engine's crossing rules for equality and missing
values. These differ from the earlier examples' NumPy
comparison, so migrated strategies can produce different trades at those
boundaries.

Compile one custom expression once and evaluate it against any same-shaped
fields:

```python
trend = phasor.Expression.compile("$close / Mean($close, 20) - 1")
values = trend.evaluate({"close": close})
```

Compile several named factors together to share repeated subexpressions:

```python
factors = phasor.FactorSet.compile({
    "trend": "$close / Mean($close, 20) - 1",
    "volume_ratio": "$volume / Mean($volume, 5)",
})
values = factors.evaluate({"close": close, "volume": volume})
```

Operators bind as in mathematics: `^` (or `**`) tightest and from the right,
then unary minus (`-2^2` is `-4`), `*` `/`, `+` `-`, comparisons, `&`, `|`.
Comparisons chain as in mathematics, `1 < $x < 3` meaning `1 < $x` and
`$x < 3`. Literal arguments may be arithmetic (`Mean($close, 2*5)`).

`phasor.FactorGrid` compiles expressions with `{name}` placeholders for every
case of a parameter grid into one graph, so what no parameter touches is
computed once and one pass evaluates every case:

```python
grid = phasor.FactorGrid.compile(
    {"fast": "Mean($close, {f})", "slow": "Mean($close, {s})"},
    phasor.params.grid(f=[5, 10], s=[20, 50], where=lambda p: p["f"] < p["s"]),
).with_warmup()
v = grid.evaluate({"close": close})            # each (rows, cases * symbols)
pf = phasor.from_signals(close, phasor.crossed_above(v["fast"], v["slow"]),
                         phasor.crossed_above(v["slow"], v["fast"]),
                         column_map=grid.column_map(close.shape[1]))
```

Keyword arguments form the full product (`FactorGrid.compile(f, n=[10, 20])`);
`grid.cases` lists each case's values in column order.

`phasor.alpha158()` and `phasor.alpha360()` return the built-in Qlib factor sets and
require explicit `open`, `high`, `low`, `close`, `volume`, and `vwap` fields.
`phasor.alpha158(config)` takes Qlib's `Alpha158DL.get_feature_config` dict
(`kbar`, `price`, `volume`, `rolling` with `windows` / `include` / `exclude`),
and `phasor.alpha158_features(config)` lists the `(name, expression)` pairs it
compiles. `phasor.qlib_label("close")` is Qlib's label
`Ref($close, -2)/Ref($close, -1) - 1`, a training target (`uses_future`).

Qlib's high-frequency operators (`DayCumsum`, `DayLast`, `Date`, `FFillNan`,
`BFillNan`, `IsNull`, `IsInf`, `Select`, `Cut`), `ChangeInstrument` and
point-in-time `P($$field)` / `PRef` read more than the panel's fields; pass it
to `evaluate` (and `evaluate_batches`, `to_parquet`) as keywords:

```python
values = factors.evaluate(
    {"close": close, "volume": volume},
    index=bar_times,                       # datetime64 or a pandas DatetimeIndex
    symbols=["600000.SH", "600036.SH"],    # ChangeInstrument looks names up here
    instruments={"SH000300": {"close": index_close}},   # outside the panel
    pit=records,                           # columns symbol, field, date, period, value
)
```

`factors.needs_timestamps` and `factors.instrument_names` say which of these a
set reads.

Qlib's windows give a value from the first row (`min_periods=1`), so a 20-row
mean on row 1 is the mean of two bars. `factors.lookbacks` says how many rows
each factor reaches back, and `factors.with_warmup()` returns the same set
evaluating to `NaN` until each symbol's windows have filled (counted from the
symbol's own first value), with cross-sectional operators skipping symbols
still warming up. A `FactorSet` keeps Qlib's values by default; the signal
paths turn warm-up on: `phasor.factors(...)`, TOML runs (`warmup = false`
under `[backtest]` turns it off) and the MCP `run_backtest` tool (option
`warmup`). There a `NaN` entry or exit is no signal.

`phasor.processors` has Qlib's data processors with Qlib's names and
arguments (`ProcessInf`, `Fillna`, `ZScoreNorm`, `RobustZScoreNorm`,
`MinMaxNorm`, `CSZScoreNorm`, `CSRankNorm`, `CSZFillna`, `TanhProcess`,
`DropnaLabel`, `DropnaProcessor`, `DropCol`, `FilterCol`) over the
`{name: (rows, symbols)}` mapping `evaluate` returns. `process` runs a list
the way Qlib's `DataHandlerLP` does; the fitted ones learn their statistics
from `fit_start_time` to `fit_end_time` only:

```python
from phasor import processors as P

data = {**phasor.alpha158().evaluate(fields), **phasor.qlib_label().evaluate(fields)}
infer = [P.RobustZScoreNorm("2015-01-01", "2019-12-31", fields_group="feature"),
         P.Fillna(fields_group="feature")]
data = P.process(data, infer, index=dates)
train = P.process(data, [P.DropnaLabel(), P.CSRankNorm(fields_group="label")])
```

A row Qlib would drop stays in the panel as `NaN` in every field.
Alpha360 returns the raw 60-row normalized history without Qlib's later data
handler processing. Custom field names such as `$funding_rate` work the same
way. The DSL supports arithmetic, comparisons, `If`, `Ref`, fixed and
expanding windows, regression, correlation/covariance, ranks, moments, WMA,
and EMA.

Expressions do not decide execution timing. Delay a close-derived factor
before turning it into a signal that fills on a later bar:

```python
entries = np.zeros_like(values["trend"], dtype=bool)
entries[1:] = values["trend"][:-1] > 0
```

## Local data and parameter sweeps

Install `pip install -e 'crates/phasor-py[data]'` for local CSV/Parquet sources.
NumPy-only backtests still work without PyArrow. `phasor.data` owns reading,
validation, BarPanel construction and auxiliary alignment, in Python: pyarrow
feeds Arrow batches to the caller's decoder, and NumPy assembles the panel and
answers the asof/window views.

`InstrumentId.parse("ADAUSDC.BINANCE")` separates `symbol` and `venue`.
`BarType.parse("ADAUSDC.BINANCE-1-MINUTE-LAST-EXTERNAL")` additionally preserves
step, aggregation, price type and source. Namespace names do not identify bars:
a successfully parsed source BarType does. `prepare()` requires the five
float64 OHLCV fields and ignores additional decoded fields.

A `DataSchema(name, input, output, decode, event_time=..., available_time=...,
instrument=...)` applies identically to all readers. `input` and `output` are
PyArrow schemas; `decode(batch, context)` returns a PyArrow RecordBatch. Input
fields select Parquet columns and supply CSV types (CSV must have a header;
column order and extra columns do not matter). Output fields are validated by
name, dtype and nullability. The decoder owns vendor encodings and units.
`context` contains `key`, `path`, `instrument`, and parsed `bar_type` when present.
The two time columns must contain non-null uint64 UTC nanoseconds. They can be
the same column. `instrument="source"` uses the source key;
`instrument=("symbol", "venue")` uses decoded string columns; `instrument=None`
means global data, broadcast only when an aligned view is requested.

```python
# One path is all a source needs: a file, a directory of one key's files, or a
# directory whose children are named after the keys.
panel = phasor.data.load("/data/crypto_bar", bar_schema, bar_types).prepare()
panel = phasor.data.load(f"/data/crypto_bar/{bar_types[0]}", bar_schema).prepare()
funding = phasor.data.load("/data/crypto_funding_rate", funding_schema, instruments)
fields = {
    "close": panel.close,
    "funding_rate": funding.asof(panel, field="rate"),
    "news_sentiment": funding.window(panel, field="sentiment", window="30m",
                                     agg="mean", empty=0.0),
}
```

A `Source` binds files to a decoder when the read should be deferred or repeated:

```python
source = phasor.data.Source("/data/crypto_bar", bar_schema, bar_types)
panel = source.prepare(start=start_ns, end=end_ns)
# The same mapping form names the keys and paths explicitly.
source = phasor.data.Source({bar_types[0]: "ada.csv", bar_types[1]: "eth.csv"}, bar_schema)
```

With one path, the files are found like this, per key, first match wins: the
path is a file; `path/<key>` is a file; `path/<key>` is a directory, whose data
files are read in filename order; or the path is a directory holding one key's
files, which works for a single key only. A directory whose children are named
after the keys has no instrument of its own, so it needs `instruments`; a single
key's directory names it, and a file name is never read as data. Only content
decides a format, so gzip is transparent (`.csv.gz`, or gzip under any name) and
a wrongly named file still reads correctly. Gzipped Parquet is refused: Parquet
compresses internally and its reader needs to seek. Layouts such as
`<root>/data/<schema>/<instrument>/` are the caller's: pass `<root>/data/<schema>`
as the path and the instrument list, and nothing about that convention enters
phasor.

`start` and `end` on `phasor.data.load()` are uint64 UTC nanoseconds or ISO-8601
strings; `Source.load()` and `Source.prepare()` take nanoseconds only. Either way
they bound the decoded availability column, half open, and never a file name:
history an asof or window view needs must be inside the range.

`BarPanel.time` is the sorted union of available bar timestamps. Missing cells
remain NaN; duplicate timestamps within one series raise an error. Each OHLCV
field has Fortran-contiguous shape `[T,S]`; `bar_types` and `instruments` describe
its columns. Rolling factor windows count panel rows, including gaps, so choose
the decoder's availability clock deliberately when feeds have different clocks.
`Dataset.batches` exposes `(DecodeContext, RecordBatch)` pairs including text.

`asof` uses the latest available row at or before panel time; equal timestamps
use the last row in stable input order, and missing history produces NaN.
`window` includes `(t-window,t]`: sum/mean skip null/nonfinite observations and
use `empty` when none remain; count counts all rows and returns zero for an
empty window. Both views align by complete instrument identity, never bare
symbol. Funding is a factor input, not an automatic funding cash flow.

Both simulation entry points accept `column_map`. Market OHLC arrays retain
`[T,S]`; signals, orders, fees, sizes and other trading parameters use `[T,C]`,
where `C=len(column_map)`. For two instruments and four parameter sets:

```python
pf = phasor.from_signals(panel.close, entries, exits,
    column_map=[0, 1, 0, 1, 0, 1, 0, 1], threads=4)
assert pf.n_cols == 8
assert pf.price_nbytes == panel.close.nbytes
# Add cash_sharing=True, group_lens=[2]*4 for four two-instrument accounts.
```

The portfolio owns one source-price copy and the integer map. Execution and
statistics both read through the map; prices are never tiled by parameter.
Factor computation remains the existing FactorSet implementation. Threads split
independent simulation accounts; rows inside each account remain sequential.

Runnable examples (all except explicit fetch can run offline):

```sh
python examples/local_data.py              # one path: file, instrument directory, schema directory
python examples/multi_data.py --source catalog
python examples/multi_data.py --source parquet
python examples/multi_data.py --source csv
python examples/crypto.py --demo
python examples/sweep.py
python examples/portfolio.py
# Full BTC/ETH/SOL/BNB USD-M history from Binance Vision:
python examples/download_binance_vision.py
# Optional real Binance download followed by read-only backtesting:
python examples/fetch.py --symbols BTCUSDT.BINANCE,ETHUSDT.BINANCE --interval 1d --from 2021-01-01
python examples/crypto.py --symbols BTCUSDT.BINANCE,ETHUSDT.BINANCE --interval 1d
```

`multi_data.py` supplies complete decoders and deterministic temporary samples;
`--dir` switches to existing files without writing samples. It prints stage
timings and source/expanded price bytes. Close-derived signals execute one row
later. Synthetic examples use `.SIM` venues or explicitly synthetic Binance
sample data.

Migration: `phasor.data.load_cached/align/fetch_*` have been removed, and so
have `phasor.data.Catalog` and `phasor.data.files`: `Catalog(root, schemas=[...])
.source(name, keys=keys)` becomes `Source(root / "data" / name, schema, keys)`,
and `files({key: path}, schema=schema)` becomes `Source({key: path}, schema)`.
Download/resume/write helpers and interval annualization now live
only in `examples/binance.py`. New fetch filenames include the venue, e.g.
`BTCUSDT.BINANCE-1d.csv`; old files may be passed directly to `Source` with an
explicit full BarType key. There is no automatic rename or cache migration.

## Build

```sh
python -m pip install --upgrade 'maturin>=1.9.4,<2.0'
maturin develop --release -m crates/phasor-py/Cargo.toml     # into the active venv
```

`abi3-py39`, so one wheel covers CPython 3.9 and up.

## Tests

```sh
python -m pip install -e 'crates/phasor-py[test,data]'
pytest crates/phasor-py/tests -q
```

The binding tests cover layout, dtypes, bounds, error messages, thread
determinism and the log join, on small cases whose fills are worked out by hand.

Shared accounts add 48 committed golden cases for both entry points, including
serial/parallel execution, all log fields, account series and statistics, in
`tests/fixtures/shared_accounts.json`. They run offline. See
[the fixture notes](../../tests/generators/README.md).

`cargo test -p phasor-py` covers the string-to-enum mappings, which are tested one
variant at a time because a table that maps `"targetvalue"` to `TargetAmount`
compiles, runs, and produces a merely different backtest.

## The one thing to know

**The engine is column-major; NumPy is row-major by default.** A `(3, 2)` array
of `0..6` is `[0,1,2,3,4,5]` in C order and `[0,2,4,1,3,5]` in F order, so
reading a C-ordered array as column-major silently backtests the *transposed*
price series — which runs, fills orders, and reports an ordinary-looking result.

Neither contiguity nor shape catches it: a C-ordered array is contiguous, and on
a square matrix the transposed reading has the same shape. So the Rust layer
refuses a C-ordered 2-D array and names the fix, and the Python shell calls
`np.asfortranarray` first — free when the array is already F-ordered, a visible
copy when it is not. The guard lives in Rust rather than only in the shell
because `phasor._phasor` is importable.

## What is exposed

Both vectorized loops — `from_signals` and `from_orders` — the `Portfolio` they
return, its metrics, its statistics table, and its order, trade, log and cash
records as NumPy arrays. `Expression`, `FactorSet`, `alpha158()`, and
`alpha360()` expose the Rust factor-expression engine through the same NumPy
boundary.
`phasor.data` supplies Binance Spot download, CSV caching and aligned OHLCV arrays
in Python.

The portfolio also answers how much was at work and how each trade went:
`pf.allocations()`, `pf.long_exposure()`, `pf.short_exposure()`,
`pf.net_exposure()` and `pf.gross_long_short()` return whole `(n_bars, …)`
matrices — per symbol column for allocations, per account for the rest —
and `pf.position_coverage()` the share of bars each column held a position.
`pf.excursions(high, low)` gives each trade's MAE and MFE, aligned with
`pf.trades()`, and `pf.edge_ratio(high, low)` one edge ratio per account. A
portfolio keeps only its closes, so the bar range is passed in; without it
both are measured on closes, which can only understate them.

`from_signals` takes the four stop modes beside `sl_stop`/`tp_stop`:
`stop_entry_price` (`close`, `fillprice`, `price`, `valprice`),
`stop_exit_price` (`stoplimit`, `stopmarket`, `price`, `close`),
`upon_stop_exit` (`close`, `reverse`, `closereduce`, `reversereduce`) and
`upon_stop_update` (`override`, `overridenan`, `keep`). `None` keeps the
engine's default.

Research tools sit in their own modules, beside the backtest rather than in it:

```python
from phasor import events, labels, robust, signals, synthetic

close = synthetic.gbm((1000, 20), mu=0.0, seed=1)    # a no-edge control
entries, exits = signals.random_entries_exits(close.shape, 10, seed=2)

study = events.forward_paths(close, entries, before=5, after=20)
study.mean_path(), study.hit_rate(20)

y, decided_at = labels.triple_barrier(close, up=0.03, down=0.02, max_hold=20)

r = returns_of_a_sweep                               # (n_bars, n_trials)
robust.pbo_cscv(r, n_splits=16)
robust.deflated_sharpe(r[:, best], per_period_sharpes)
```

Judging a strategy past its Sharpe:

```python
from phasor import allocation, predictions, returns, robust, validation

returns.metric(r, "hit_ratio")                         # share of moving bars that gained
returns.metric(r, "return_concentration", cutoff=0.05) # share of gains from the best 5% of bars
returns.stats(r)["max_drawdown_recovery"]              # bars from the deepest valley back to its peak
robust.bootstrap(r, n=1000, block=5)                   # drawdown and Sharpe over resampled paths

validation.position_delay_test(held, asset_returns, delays=(0, 1, 2, 3), cost=0.0007)
validation.permutation_test(held, asset_returns, statistic="sharpe", block=6)   # 2-D: one shuffle for all assets

predictions.power(signal, forward_returns)                        # R², quantile returns, monotonicity
predictions.power(factor, forward_returns, axis="cross_section")  # a factor's IC and ICIR
predictions.decay(signal, close, horizons=(1, 4, 16))             # return in the signal's direction, h bars on

allocation.marginal_contribution(book_returns, candidate)  # correlations and the book's Sharpe gain
```

`events` and `labels` read the future on purpose and must never feed a
simulation. `signals` and `synthetic` draw from `(seed, row, col)`, so a seed
reproduces and a column does not change when others are added. Sharpe ratios
in `robust` are per period, not annualized.

`phasor.rates.per_bar` turns annual rates into the per-bar rates the holding
charges take.

### Logs

`log=True`, or a `(n_bars, n_cols)` mask to log one column of a sweep. Then
`pf.logs()` for the 38-field schema as arrays, `pf.log_reasons()` for the outcome
of each row in words, and `pf.reason_counts(col)` for the tally — which is the
first thing to reach for when a column traded far less than expected, because
there is almost always one reason dominating.

One row per order **considered**, filled or not. `from_orders` submits an order
per element and so logs every element; `from_signals` reaches the kernel only
when the signals resolved to a size, so a quiet bar gets no row. `order_id` joins
a row to the fill it produced, or is `-1`.

Logging is additional, never alternative: the same workload produces identical
order records with it on and off, and a test asserts that.

## Size types and direction in the two loops

Both loops accept all six size types, but a *target* size reads differently in
each. In `from_orders` it says where the position should end up, which is
unambiguous per element. In `from_signals` it is paired with a signal that only
says which way to move: an entry moves the position to the target, from flat,
from the other side, or, where `accumulate` lets it add, from the same side, and
an exit closes it.

`direction` defaults differently: `both` for orders, where the size's sign
already says which way it goes, and `longonly` for signals.

## Two traps this layer inherits

**A column that never traded has an infinite Sharpe.** Zero return over zero
volatility. That is the defined answer, and it tops a naive `argmax` over a
sweep. `pf.traded_only(values)` masks those columns to `NaN`, which every
reduction skips.

**Each individual metric accessor rebuilds the derivation chain.** The engine's
`Portfolio` borrows its prices and caches its derivations; a `#[pyclass]` cannot
hold a borrow of itself, so the portfolio is rebuilt per call and the cache
starts empty each time. `pf.metrics()` and `pf.stats()` exist as bulk calls for
that reason — six scalars pulled one at a time is six rebuilds.
