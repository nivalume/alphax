# Alphax Strategy & Feature Examples

This directory contains self-contained, runnable example scripts demonstrating **every feature, scenario, and dimension** of the `alphax` backtesting and quantitative research engine.

## Quick Start

All strategy and research examples generate synthetic market data out of the box (no API keys or network connection required) and can be run directly with `uv`:

```bash
uv run python examples/factors_and_alphas.py
uv run python examples/indicators_and_patterns.py
uv run python examples/stops_limits_and_risk.py
uv run python examples/margin_futures_and_carry.py
uv run python examples/allocation_and_intraday.py
uv run python examples/stateful_callbacks.py
uv run python examples/research_and_validation.py
uv run python examples/live_scale_and_ops.py
```

---

## Example Matrix by Dimension & Scenario

### 1. Core Strategy & Portfolio Basics
| Script | Scenario & Features Covered |
| :--- | :--- |
| [`sweep.py`](./sweep.py) | `@alphax.strategy` decorator, `alphax.factors`, `crossed_above` / `crossed_below`, single backtest `.run()`, and Cartesian parameter `.sweep()`. |
| [`portfolio.py`](./portfolio.py) | Target-percent rebalancing (`from_orders(..., size_type="targetpercent", cash_sharing=True)`) and multi-group allocation (`group_lens`, `column_map`). |
| [`multi_data.py`](./multi_data.py) | Multi-frequency data alignment: `DataSchema`, `Catalog`, `BarPanel`, point-in-time `Dataset.asof` joins, and rolling `Dataset.window` tensor views. |

### 2. Factor Engine, Alpha Libraries & Technical Analysis
| Script | Scenario & Features Covered |
| :--- | :--- |
| [`factors_and_alphas.py`](./factors_and_alphas.py) | `Expression.compile` (`required_fields`, `uses_future`), full Qlib-style DSL (time-series, elementwise, `IndNeutralize`, `CSResi`, `CSRank`, `Scale`), built-in `alpha158()`, `alpha360()`, `wq101(all=True)`, `evaluate_batches`, `to_parquet`, and a **Cross-Sectional Market-Neutral Long/Short Strategy** with exposure analytics. |
| [`indicators_and_patterns.py`](./indicators_and_patterns.py) | All 8 Rust indicators (`ma`, `mstd`, `bbands`, `rsi`, `stoch`, `macd`, `atr`, `obv` + `true_range`), custom Python `IndicatorFactory`, live bar-by-bar `indicators.Stream`, and a **Multi-Window Chart Pattern Search Strategy** (`patterns.search`, `patterns.signals`). |

### 3. Execution Realism, Order Types, Risk Rules & Derivatives
| Script | Scenario & Features Covered |
| :--- | :--- |
| [`stops_limits_and_risk.py`](./stops_limits_and_risk.py) | `price="nextopen"`, bid/ask spreads, lot granularity, partial fills, audit logs (`pf.logs()`, `pf.reason_counts()`), resting pullback & breakout limit orders (`order_type="limit"`, `limit_reverse`, `limit_tif`), multi-tranche stop ladders (`stop_ladder="adaptweighted"`), `td_stop` / `dt_stop`, Rust risk rules (`risk`, `risk_stop`, `breakeven`, `max_positions`, `max_exposure`), and MFE/MAE `excursions` / `edge_ratio`. |
| [`margin_futures_and_carry.py`](./margin_futures_and_carry.py) | Cash leverage (`lazy` vs `eager`, `debt_rate`, `borrow_rate`), **Spot + Perpetual Delta-Neutral Funding Rate Carry** (`account_type=["cash", "margin"]`, `funding_rate`, `mark_price`), tiered maintenance margin & partial liquidation (`liquidation_step`, `auto_deleverage`), continuous futures roll (`alphax.futures`), and FX conversion (`alphax.fx`). |

### 4. Portfolio Construction, Intraday Sessions & Path-Dependent Callbacks
| Script | Scenario & Features Covered |
| :--- | :--- |
| [`allocation_and_intraday.py`](./allocation_and_intraday.py) | Dynamic asset allocation (`equal_weight`, `inverse_volatility`, `min_variance`, `max_sharpe`, `risk_parity`) with DCA `cash_deposits`, calendar & signal utilities (`alphax.signals`), DST-aware intraday trading sessions (`alphax.sessions`) with EOD `force_exit`, and multi-resolution `1h -> 15m` alignment (`alphax.resolution`). |
| [`stateful_callbacks.py`](./stateful_callbacks.py) | Path-dependent Python hooks: `from_order_func` (`pre_group_func`, `pre_segment_func`, `order_func`, `post_order_func`, `post_segment_func` drawdown circuit breaker), `flexible=True` multi-leg statistical arbitrage pairs trading, and `from_signals` (`signal_func`, `adjust_func` ratcheting stops). |

### 5. Quantitative Research, ML Validation & Production Ops
| Script | Scenario & Features Covered |
| :--- | :--- |
| [`research_and_validation.py`](./research_and_validation.py) | Synthetic markets (`alphax.synthetic`), event study forward paths (`alphax.events`), ML triple-barrier & trend/pivot labeling (`alphax.labels`), classifier evaluation (`alphax.predictions`), constrained grid/random search (`alphax.params`), walk-forward & purged K-fold CV (`alphax.validation`), and anti-overfitting corrections (`probabilistic_sharpe`, `deflated_sharpe`, `pbo_cscv`). |
| [`live_scale_and_ops.py`](./live_scale_and_ops.py) | Live bar-by-bar `Stepper` (`propose`, `commit`, `reconcile`), incremental continuation (`resume=pf`), portfolio algebra (`alphax.portfolios`) & standalone `alphax.returns`, `chunked.run`, `distributed.run`, disk `Cache`, `save`/`load`, declarative TOML `alphax.config`, `alphax.schedule`, and `check_shapes` / `docs.search`. |

### 6. Live Exchange Data & Webhook Notifications (Network Optional)
| Script | Scenario & Features Covered |
| :--- | :--- |
| [`crypto.py`](./crypto.py), [`binance.py`](./binance.py), [`fetch.py`](./fetch.py), [`download_binance_vision.py`](./download_binance_vision.py) | Fetching & caching OHLCV bars from Binance Spot REST API and Binance Vision S3 archives. |
| [`notifiers.py`](./notifiers.py) | Sending trade & rebalance alerts via `SlackNotifier` and `TelegramNotifier`. |
