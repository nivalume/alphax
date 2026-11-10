"""Factor Engine, Built-in Alpha Libraries, and Cross-Sectional Long/Short Strategy.

Scenario & Dimensions Covered:
  1. Single-Expression Compilation & Introspection:
     - `Expression.compile`, `required_fields`, `uses_future` look-ahead detection.
  2. Rich Qlib-Style Factor DSL across 3 Operator Dimensions:
     - Time-series rolling: `Ref`, `Delta`, `Mean`, `Std`, `Skew`, `Kurt`, `Max`, `Min`,
       `Med`, `Mad`, `Rank`, `Quantile`, `IdxMax`, `IdxMin`, `Slope`, `Rsquare`, `Resi`,
       `Corr`, `Cov`, `WMA`, `EMA`, `DecayLinear`, `Count`, `Prod`.
     - Elementwise & conditional: `If`, `Clip`, `FillNa`, `Mask`, `SignedPower`, `Abs`,
       `Sign`, `Log`.
     - Cross-sectional & group neutralization: `CSRank`, `CSZScore`, `CSDemean`,
       `IndNeutralize`, `CSResi`, `Scale`.
  3. Built-in Academic & Industry Factor Libraries:
     - Qlib `alpha158()` and `alpha360()`.
     - WorldQuant `wq101(all=False)` (82 price-volume alphas) and `wq101(all=True)`
       (all 101 alphas including `cap` and `IndNeutralize` sector/industry/subindustry).
     - `wq101_formula(k)` inspection.
  4. Out-of-Core / Memory-Bounded Evaluation:
     - `FactorSet.evaluate_batches` and `FactorSet.to_parquet`.
  5. Cross-Sectional Market-Neutral Long/Short Factor Strategy:
     - Ranking assets each day by composite industry-neutral alpha, going long the top
       quintile and short the bottom quintile in a shared cash pool (`direction="both"`,
       `cash_sharing=True`, `size_type="targetpercent"`).
     - Portfolio exposure analytics (`allocations`, `long_exposure`, `short_exposure`,
       `net_exposure`, `gross_long_short`, `position_coverage`).

Run:
    uv run python examples/factors_and_alphas.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import alphax


def make_cross_sectional_universe(
    n_bars: int = 360,
    n_assets: int = 15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a synthetic multi-asset OHLCV + fundamentals + industry panel."""
    rng = np.random.default_rng(seed)

    # 3 industries (0, 1, 2) with 5 assets each
    industry_ids = np.repeat(np.arange(3, dtype=np.float64), n_assets // 3)
    industry = np.broadcast_to(industry_ids, (n_bars, n_assets)).copy()
    sector = np.floor(industry / 2.0)
    subindustry = np.tile(np.arange(n_assets, dtype=np.float64), (n_bars, 1))

    # Common market factor + sector factor + idiosyncratic returns with slight momentum
    mkt_ret = rng.normal(0.0003, 0.008, size=(n_bars, 1))
    ind_ret = rng.normal(0.0, 0.005, size=(n_bars, 3))[:, industry_ids.astype(int)]
    idio_ret = rng.normal(0.0, 0.012, size=(n_bars, n_assets))
    total_ret = mkt_ret + ind_ret + idio_ret

    close = 100.0 * np.exp(np.cumsum(total_ret, axis=0))
    open_ = close * np.exp(rng.normal(0.0, 0.003, size=(n_bars, n_assets)))
    spread = np.abs(rng.normal(0.008, 0.003, size=(n_bars, n_assets)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=12.0, sigma=0.5, size=(n_bars, n_assets))
    vwap = (high + low + close) / 3.0
    cap = close * rng.uniform(1e6, 5e7, size=(1, n_assets))

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
        "cap": cap,
        "sector": sector,
        "industry": industry,
        "subindustry": subindustry,
    }


def demo_expression_introspection(fields: dict[str, np.ndarray]) -> None:
    print("=== 1. Expression Compilation & Look-Ahead Introspection ===")
    expr = alphax.Expression.compile("CSRank(Mean($close / Ref($close, 1) - 1, 20))")
    future_expr = alphax.Expression.compile("Ref($close, -5) / $close - 1")

    vals = expr.evaluate(fields)
    print(f"Compiled:        {expr}")
    print(f"Required fields: {expr.required_fields}")
    print(f"Uses future?     past_expr={expr.uses_future}, future_expr={future_expr.uses_future}")
    print(f"Output shape:    {vals.shape}, last-row range=[{np.nanmin(vals[-1]):.2f}, {np.nanmax(vals[-1]):.2f}]")


def demo_custom_factor_dsl(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    print("\n=== 2. Rich Multi-Dimensional FactorSet DSL ===")
    fset = alphax.FactorSet.compile(
        {
            # Time-series momentum, regression slope/R2/residual, and decay
            "ts_mom_10": "$close / Ref($close, 10) - 1",
            "ts_linreg_slope": "Slope($close, 20) / $close",
            "ts_linreg_r2": "Rsquare($close, 20)",
            "ts_linreg_resi": "Resi($close, 20) / Std($close, 20)",
            "ts_decay_flow": "DecayLinear(($close - $open) * Log($volume + 1), 10)",
            # Higher moments, robust dispersion, and quantile positioning
            "ts_skew_kurt": "Skew($close / Ref($close, 1) - 1, 20) - 0.1 * Kurt($close / Ref($close, 1) - 1, 20)",
            "ts_robust_z": "($close - Med($close, 20)) / (Mad($close, 20) + 1e-8)",
            "ts_pos_in_range": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-8)",
            "ts_arg_spread": "(IdxMax($high, 20) - IdxMin($low, 20)) / 20.0",
            "ts_pv_corr": "Corr($close / Ref($close, 1) - 1, Log($volume + 1), 15)",
            # Elementwise & conditional regimes
            "cond_reversal": "If($close > EMA($close, 20), -Delta($close, 3) / $close, Delta($close, 3) / $close)",
            "clipped_power": "Clip(SignedPower(CSZScore($close / $vwap - 1), 1.5), -3.0, 3.0)",
            # Cross-sectional neutralization & orthogonalization
            "ind_neutral_mom": "IndNeutralize($close / Ref($close, 20) - 1, $industry)",
            "size_ortho_mom": "CSResi($close / Ref($close, 20) - 1, Log($cap))",
            # Composite market-neutral alpha scaled so sum(abs(w)) == 1 each bar
            "composite_alpha": (
                "Scale(CSDemean("
                "  0.40 * CSRank(IndNeutralize($close / Ref($close, 15) - 1, $industry))"
                "  - 0.35 * CSRank(IndNeutralize($close / Ref($close, 3) - 1, $industry))"
                "  - 0.25 * CSRank(Corr($close, $volume, 10))"
                "))"
            ),
        }
    )
    print(f"FactorSet count: {len(fset.names)} factors | required_fields={fset.required_fields}")
    out = fset.evaluate(fields)
    for name in ("ts_linreg_slope", "ind_neutral_mom", "size_ortho_mom", "composite_alpha"):
        arr = out[name]
        print(f"  {name:<18} nan_ratio={np.isnan(arr).mean():.1%}  last_abs_sum={np.nansum(np.abs(arr[-1])):.4f}")

    # One-step convenience evaluation via `alphax.factors(data, **named_exprs)` -> returns a dict
    quick = alphax.factors(fields, fast_ema="EMA($close, 8)", slow_ema="EMA($close, 21)")
    print(f"Quick `alphax.factors(fields, ...)` keys={list(quick.keys())}, fast_ema shape={quick['fast_ema'].shape}")
    return out


def demo_builtin_alpha_libraries(fields: dict[str, np.ndarray]) -> None:
    print("\n=== 3. Built-in Alpha Libraries (Alpha158, Alpha360, WorldQuant 101) ===")
    a158 = alphax.alpha158()
    a360 = alphax.alpha360()
    wq_pv = alphax.wq101(all=False)
    wq_all = alphax.wq101(all=True)

    out_158 = a158.evaluate(fields)
    out_wq = wq_all.evaluate(fields)
    print(f"Alpha158:     {len(a158.names):>3} factors evaluated (e.g. KMID, ROC5, RSV20, WVMA60)")
    print(f"Alpha360:     {len(a360.names):>3} factors compiled (60 lags x 6 OHLCV/VWAP fields)")
    print(f"WQ101 (pv):   {len(wq_pv.names):>3} price-volume factors")
    print(f"WQ101 (all):  {len(wq_all.names):>3} total factors (including IndNeutralize & cap)")
    print(f"Sample WQ#48: {alphax.wq101_formula(48)}")
    print(f"Sample WQ#101:{alphax.wq101_formula(101)} -> mean={np.nanmean(out_wq['alpha101']):.4f}")
    print(f"Sample A158['KLEN'] mean={np.nanmean(out_158['KLEN']):.4f}")

    # Out-of-core symbol-batched evaluation (`evaluate_batches`) & Parquet export (`to_parquet`)
    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = Path(tmpdir) / "custom_factors.parquet"
        sub_fset = alphax.FactorSet.compile({"roc5": "$close / Ref($close, 5) - 1", "vol5": "Std($close, 5)"})
        col_spans: list[tuple[int, int]] = []
        sub_fset.evaluate_batches(fields, batch=5, sink=lambda s, e, vals: col_spans.append((s, e)))
        sub_fset.to_parquet(str(pq_path), fields, batch=5)
        print(f"Symbol-batched eval spans (batch=5 cols): {col_spans} | Parquet written: {pq_path.stat().st_size} bytes")


def demo_cross_sectional_long_short_strategy(
    fields: dict[str, np.ndarray],
    factor_out: dict[str, np.ndarray],
) -> None:
    print("\n=== 4. Cross-Sectional Market-Neutral Long/Short Strategy ===")
    close = fields["close"]
    alpha = factor_out["ind_neutral_mom"]
    n_bars, n_assets = close.shape

    # Rank cross-sectionally at each bar after warm-up; go long top 3 (+12% each), short bottom 3 (-12% each)
    cs_rank = alphax.Expression.compile("CSRank(IndNeutralize($close / Ref($close, 15) - 1, $industry))").evaluate(fields)
    warmup = 20

    # Rebalance weekly (every 5 bars) using signed targetpercent orders (`np.nan` = hold on non-rebalance bars)
    target_weights = np.full((n_bars, n_assets), np.nan)
    for t in range(warmup, n_bars, 5):
        row_rank = cs_rank[t]
        w = np.zeros(n_assets, dtype=np.float64)
        w[row_rank >= 0.80] = +0.12
        w[row_rank <= 0.20] = -0.12
        target_weights[t] = w

    pf = alphax.from_signals(
        close,
        order_mode=True,
        size=target_weights,
        size_type="targetpercent",
        direction="both",
        cash_sharing=True,
        call_seq="auto",  # sell/reduce before buying on rebalance bars
        init_cash=100_000.0,
        fees=0.0005,
        from_ago=1,  # trade on the bar after signal observation
    )

    alloc = pf.allocations()
    cov = pf.position_coverage()
    print(
        f"Total Return:      {pf.total_return()[0]:.2%}  | Sharpe: {pf.sharpe_ratio()[0]:.2f}  "
        f"| Max DD: {pf.max_drawdown()[0]:.2%}"
    )
    print(
        f"Avg Long Exposure: {pf.long_exposure().mean():.2%}  "
        f"| Avg Short Exposure: {pf.short_exposure().mean():.2%}  "
        f"| Avg Net Exposure: {pf.net_exposure().mean():+.2%}"
    )
    print(
        f"Max Gross L/S:     {pf.gross_long_short().max():.2%}  "
        f"| Mean Asset Coverage: {cov.mean():.1%}  "
        f"| Total Trades: {pf.trade_count()[0]} (Fills: {pf.n_fills})"
    )
    _ = (alpha, alloc)


def main() -> None:
    fields = make_cross_sectional_universe()
    demo_expression_introspection(fields)
    factor_out = demo_custom_factor_dsl(fields)
    demo_builtin_alpha_libraries(fields)
    demo_cross_sectional_long_short_strategy(fields, factor_out)


if __name__ == "__main__":
    main()
