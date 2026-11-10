"""Event Studies, ML Triple-Barrier Labeling, Walk-Forward CV & Anti-Overfitting.

Scenario & Dimensions Covered:
  1. Synthetic Market Generation (`alphax.synthetic`) & Null-Hypothesis Random Signals:
     - `synthetic.gbm`, `synthetic.random_walk`, `synthetic.ohlc_from_close`.
     - `signals.random_n`, `signals.random_by_prob`, `signals.random_entries_exits`,
       `signals.random_like`.
  2. Event Study Forward Price Paths (`alphax.events.forward_paths`):
     - Analyzing post-event drift (`mean_path()`, `median_path()`, `hit_rate(offset)`, `paths`)
       around RSI oversold crossbars before committing to a strategy rule.
  3. ML Labeling (`alphax.labels`) & Prediction Evaluation (`alphax.predictions`):
     - `labels.triple_barrier`, `labels.fixed_horizon_returns`, `labels.trend_labels`,
       `labels.pivot_labels`, `labels.mean_labels`.
     - Evaluating directional classifiers against `"hold"` and `"random"` baselines via
       `predictions.backtest` and `predictions.compare`.
  4. Constrained Grid & Random Search (`alphax.params`):
     - `params.grid(where=...)`, `params.random` (`IntUniform`, `Uniform`, `LogUniform`,
       `Choice`), parallel `params.run(n_jobs=2)`, and `params.neighbors` plateau checks.
  5. Cross-Validation Splitters & Statistical Hypothesis Tests (`alphax.validation`):
     - `rolling`, `expanding`, `purged_kfold` (with purge & embargo), `calendar`,
       `from_ranges`, `from_masks`, and `walk_forward`.
     - `permutation_test`, `shuffled_prices`, `shuffled_test`, and `signal_delay_test`.
  6. Multiple-Testing & Selection-Bias Corrections (`alphax.robust`):
     - `probabilistic_sharpe` (PSR), `expected_max_sharpe`, `deflated_sharpe` (DSR),
       and `pbo_cscv` (Probability of Backtest Overfitting via CSCV).

Run:
    uv run python examples/research_and_validation.py
"""

from __future__ import annotations

import numpy as np
import alphax
from alphax import events, labels, params, predictions, robust, signals, synthetic, validation


def demo_synthetic_and_event_study() -> dict[str, np.ndarray]:
    print("=== 1. Synthetic OHLCV (`alphax.synthetic`) & Event Study (`alphax.events`) ===")
    close = synthetic.gbm((600, 3), start=100.0, mu=0.08, sigma=0.22, seed=42)
    rw = synthetic.random_walk((600, 1), start=100.0, step_std=0.8, seed=42)
    open_, high, low = synthetic.ohlc_from_close(close, spread=0.008, seed=42)
    ohlc = {"open": open_, "high": high, "low": low, "close": close}

    # Event: 14-bar RSI crosses below 35 (oversold)
    # Note: `crossed_below(a, b)` is expressed via `crossed_above(b, a)`
    rsi = alphax.indicators.rsi(close, window=14).rsi
    th35 = np.full_like(rsi, 35.0)
    oversold_event = alphax.crossed_above(th35, rsi)

    study = events.forward_paths(close, oversold_event, before=3, after=15)
    # Null-hypothesis random signals preserving exact signal counts per column
    rand_like = signals.random_like(oversold_event, seed=42)
    rand_n = signals.random_n(close.shape, 12, seed=42)
    rand_prob = signals.random_by_prob(close.shape, 0.03, seed=42)
    rand_en, rand_ex = signals.random_entries_exits(close.shape, 10, seed=42)

    print(
        f"Oversold Events: {len(study.idx)} | +15-bar Mean Drift: {study.mean_path()[-1]:+.2%} | "
        f"Median: {study.median_path()[-1]:+.2%} | Hit Rate (+15): {study.hit_rate(15):.1%}"
    )
    print(
        f"Random-like count parity: {oversold_event.sum(axis=0).tolist()} == {rand_like.sum(axis=0).tolist()} | "
        f"rand_n={int(rand_n.sum())}, rand_prob={int(rand_prob.sum())}, "
        f"pairs=({int(rand_en.sum())}, {int(rand_ex.sum())}) | RW shape={rw.shape}"
    )
    return ohlc


def demo_ml_labeling_and_predictions(ohlc: dict[str, np.ndarray]) -> None:
    print("\n=== 2. ML Labeling (`alphax.labels`) & Classifier Backtest (`alphax.predictions`) ===")
    close = ohlc["close"][:, 0]

    tb_label, tb_decided = labels.triple_barrier(close, up=0.025, down=0.02, max_hold=12)
    fh = labels.fixed_horizon_returns(close, horizon=5)
    tr = labels.trend_labels(close, horizon=10, threshold=0.015)
    pv = labels.pivot_labels(close, threshold=0.03)
    mn = labels.mean_labels(close, window=10, wait=1)

    # Simulate a noisy ML model that predicts the triple-barrier label with ~58% accuracy
    rng = np.random.default_rng(9)
    clean_dir = np.nan_to_num(tb_label, nan=0.0)
    flip = rng.random(clean_dir.shape) < 0.42
    model_preds = np.where(flip, -clean_dir, clean_dir)

    pfs = predictions.backtest(
        close,
        model_preds,
        enter_above=0.5,
        exit_below=0.0,
        baselines=("hold", "random"),
        seed=42,
        init_cash=10_000.0,
        fees=0.0005,
    )
    cmp = predictions.compare(
        pfs,
        metrics=("total_return", "sharpe_ratio", "max_drawdown", "trade_count"),
    )
    hold_bars = tb_decided - np.arange(len(close))
    print(
        f"Triple-Barrier Labels: +1={int((tb_label == 1).sum())}, -1={int((tb_label == -1).sum())}, "
        f"0={int((tb_label == 0).sum())} | Mean hold={float(np.mean(hold_bars)):.1f} bars"
    )
    print(
        f"Other Label Shapes: fixed_horizon={fh.shape}, trend={tr.shape}, pivots={pv.shape}, mean={mn.shape}"
    )
    print(
        f"Prediction Compare: model return={cmp['total_return']['model'][0]:+.2%}, "
        f"hold={cmp['total_return']['hold'][0]:+.2%}, random={cmp['total_return']['random'][0]:+.2%}"
    )


def demo_param_search_and_plateau(ohlc: dict[str, np.ndarray]) -> np.ndarray:
    print("\n=== 3. Parameter Grid/Random Search & Plateau Stability (`alphax.params`) ===")
    close = ohlc["close"][:, 0]

    grid_space = params.grid(
        fast=[5, 10, 15, 20],
        slow=[20, 30, 40, 60],
        where=lambda p: p["fast"] < p["slow"],
    )
    rand_space = params.random(
        6,
        seed=42,
        where=lambda p: p["fast"] < p["slow"],
        fast=params.IntUniform(4, 18),
        slow=params.IntUniform(22, 65),
        sl=params.Uniform(0.01, 0.05),
        fees=params.LogUniform(1e-4, 2e-3),
        ewm=params.Choice([False, True]),
    )

    def eval_ma(*, fast: int, slow: int) -> dict[str, float]:
        f = alphax.indicators.ma(close, fast).ma
        s = alphax.indicators.ma(close, slow).ma
        pf = alphax.from_signals(
            close,
            alphax.crossed_above(f, s),
            alphax.crossed_above(s, f),
            init_cash=10_000.0,
            fees=0.0005,
        )
        return {"sharpe_ratio": float(pf.sharpe_ratio()[0]), "total_return": float(pf.total_return()[0])}

    results = params.run(eval_ma, grid_space, n_jobs=2, executor="thread")
    best_combo = results.best(key="sharpe_ratio")
    best_i = results.params.index(best_combo)
    plateau = params.neighbors(results, key="sharpe_ratio")

    print(
        f"Grid combos (fast < slow): {len(grid_space)} | Random sample count: {len(rand_space)}"
    )
    print(
        f"Best Grid Combo: {best_combo} -> Sharpe={plateau['value'][best_i]:.2f}, "
        f"Neighbor mean={plateau['mean'][best_i]:.2f} (ratio={plateau['ratio'][best_i]:.2f})"
    )

    # Collect per-bar returns across all grid trials for CSCV / DSR analysis
    trial_returns = []
    for p in grid_space:
        f = alphax.indicators.ma(close, p["fast"]).ma
        s = alphax.indicators.ma(close, p["slow"]).ma
        pf = alphax.from_signals(
            close,
            alphax.crossed_above(f, s),
            alphax.crossed_above(s, f),
            init_cash=10_000.0,
            fees=0.0005,
        )
        trial_returns.append(pf.returns(0))
    return np.column_stack(trial_returns)


def demo_validation_and_overfitting_tests(ohlc: dict[str, np.ndarray], returns_matrix: np.ndarray) -> None:
    print("\n=== 4. Walk-Forward CV (`alphax.validation`) & Deflated Sharpe / PBO (`alphax.robust`) ===")
    close = ohlc["close"][:, 0]
    n_bars = len(close)
    index = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2024-01-01") + np.timedelta64(n_bars, "D"),
        np.timedelta64(1, "D"),
    )

    # Demonstrate all CV splitters: rolling, expanding, purged_kfold, calendar, from_ranges, from_masks
    roll_splits = validation.rolling(n_bars, train=200, test=80, step=80)
    exp_splits = validation.expanding(n_bars, min_train=200, test=80, step=80)
    pkf_splits = validation.purged_kfold(n_bars, k=5, purge=5, embargo=5)
    cal_splits = validation.calendar(index, "M", train=6, test=2, step=2)
    rng_splits = validation.from_ranges(
        index, [("2024-01-01", "2024-09-01", "2024-09-01", "2025-01-01")]
    )
    msk_splits = validation.from_masks(np.arange(n_bars) < 400, np.arange(n_bars) >= 400)

    combos = [(5, 20), (10, 30), (15, 45)]

    def score_grid(rows: np.ndarray) -> dict[tuple[int, int], float]:
        c = close[rows]
        out: dict[tuple[int, int], float] = {}
        for f_w, s_w in combos:
            f = alphax.indicators.ma(c, f_w).ma
            s = alphax.indicators.ma(c, s_w).ma
            pf = alphax.from_signals(c, alphax.crossed_above(f, s), alphax.crossed_above(s, f), init_cash=10_000.0)
            out[(f_w, s_w)] = float(pf.total_return()[0])
        return out

    wf = validation.walk_forward(score_grid, n_bars, roll_splits)

    # Signal permutation, shuffled prices, and execution delay degradation tests
    @alphax.strategy
    def ma_cross(data: dict[str, np.ndarray], fast: int = 10, slow: int = 30):
        f = alphax.indicators.ma(data["close"], fast).ma
        s = alphax.indicators.ma(data["close"], slow).ma
        return alphax.crossed_above(f, s), alphax.crossed_above(s, f)

    pf_base = ma_cross.run({"close": close}, params={"fast": 10, "slow": 30}, init_cash=10_000.0)
    asset_rets = np.r_[0.0, np.diff(close) / close[:-1]]
    positions = (pf_base.assets(0) > 0).astype(float)

    perm = validation.permutation_test(positions, asset_rets, statistic="sharpe", n=200, block=5, seed=42)
    shuf = validation.shuffled_test(
        lambda c: float(ma_cross.run({"close": c}, params={"fast": 10, "slow": 30}, init_cash=10_000.0).total_return()[0]),
        close,
        n=25,
        block=5,
        seed=42,
    )
    delay = validation.signal_delay_test(
        ma_cross,
        {"close": close},
        params={"fast": 10, "slow": 30},
        delays=(0, 1, 2, 3),
        metric="total_return",
        init_cash=10_000.0,
    )

    # Anti-overfitting metrics: PSR, Expected Max Sharpe, DSR, and PBO (CSCV)
    # Note: robust.* operates on unannualized per-bar Sharpe ratios (ddof=1)
    stds = np.std(returns_matrix, axis=0, ddof=1)
    trial_srs = np.where(stds > 0, np.mean(returns_matrix, axis=0) / stds, 0.0)
    best_idx = int(np.argmax(trial_srs))
    best_rets = returns_matrix[:, best_idx]

    psr = robust.probabilistic_sharpe(best_rets, benchmark_sr=0.0)
    sr_var = float(np.var(trial_srs, ddof=1))
    emax_sr = robust.expected_max_sharpe(returns_matrix.shape[1], sr_var)
    dsr = robust.deflated_sharpe(best_rets, trial_srs)
    pbo = robust.pbo_cscv(returns_matrix, n_splits=8)

    print(
        f"Splits: rolling={len(roll_splits)}, expanding={len(exp_splits)}, purged_kfold={len(pkf_splits)}, "
        f"calendar={len(cal_splits)}, ranges={len(rng_splits)}, masks={len(msk_splits)}"
    )
    print(
        f"Walk-Forward mean IS={np.mean([r['train_score'] for r in wf]):.2f} -> "
        f"OOS={np.mean([r['test_score'] for r in wf]):.2f}"
    )
    print(
        f"Null Tests: permutation p={perm['p_value']:.3f}, shuffled-price p={shuf['p_value']:.3f} | "
        f"Signal delay returns={{d: round(float(v[0]), 4) for d, v in delay.items()}} -> "
        f"{ {d: round(float(v[0]), 4) for d, v in delay.items()} }"
    )
    print(
        f"Robustness ({returns_matrix.shape[1]} trials): PSR={psr:.3f}, E[max SR]={emax_sr:.3f}, "
        f"DSR={dsr:.3f}, PBO (CSCV)={pbo:.2%}"
    )


def main() -> None:
    ohlc = demo_synthetic_and_event_study()
    demo_ml_labeling_and_predictions(ohlc)
    returns_matrix = demo_param_search_and_plateau(ohlc)
    demo_validation_and_overfitting_tests(ohlc, returns_matrix)


if __name__ == "__main__":
    main()
