"""Fixed evaluation of one registered factor spec: members -> composite position -> alphax backtests.

Nothing here selects on results except a pre-registered walk-forward rule, which only
uses returns before each refit point. Every run is a single test in the ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import alphax
import numpy as np

from alphax_lab import data as bars
from alphax_lab.types.factor_ts import spec as specs

BARS_PER_YEAR = 365.0 * 24.0
DAY_NS = 24 * bars.HOUR_NS
MIN_IC_SAMPLES = 24


# ---------------------------------------------------------------- factor values and positions


def evaluate(expressions: dict[str, str], fields: dict) -> dict[str, np.ndarray]:
    factor_set = alphax.FactorSet.compile(expressions)
    if factor_set.uses_future:
        raise ValueError("expression uses future data")
    values = factor_set.evaluate({name: np.asfortranarray(fields[name]) for name in factor_set.required_fields})
    out = {}
    for name, value in values.items():
        value = np.asarray(value, dtype=np.float64)
        value[~np.isfinite(value)] = np.nan
        out[name] = value
    return out


def zscore(matrices: list[np.ndarray], window: int) -> list[np.ndarray]:
    """Causal rolling z-score of several (T, S) matrices in one alphax call."""
    if not matrices:
        return []
    width = matrices[0].shape[1]
    stacked = np.asfortranarray(np.concatenate(matrices, axis=1))
    w = int(window)
    z = evaluate({"z": f"($x-Mean($x,{w}))/Std($x,{w})"}, {"x": stacked})["z"]
    z[: min(w, z.shape[0])] = np.nan
    return [z[:, i * width : (i + 1) * width] for i in range(len(matrices))]


def _ffill_state(events: np.ndarray) -> np.ndarray:
    """Forward-fills 1/0 events; NaN keeps the previous state, which starts at 0."""
    valid = ~np.isnan(events)
    index = np.where(valid, np.arange(events.shape[0])[:, None], -1)
    np.maximum.accumulate(index, axis=0, out=index)
    out = np.take_along_axis(np.where(valid, events, 0.0), np.maximum(index, 0), axis=0)
    out[index < 0] = 0.0
    return out


def position(factor, z, rule: str, params: dict, direction: int, side: str) -> np.ndarray:
    """Maps a factor to a target position in [-1, 1] per symbol."""
    if rule == "sign":
        raw = np.sign(direction * factor)
    else:
        z = direction * z
        if rule == "zscore":
            raw = np.clip(z / params["scale"], -1.0, 1.0)
        else:
            entry, exit_ = params["entry"], params["exit"]
            with np.errstate(invalid="ignore"):
                long_events = np.where(z > entry, 1.0, np.where(z < exit_, 0.0, np.nan))
                short_events = np.where(z < -entry, 1.0, np.where(z > -exit_, 0.0, np.nan))
            raw = _ffill_state(long_events) - _ffill_state(short_events)
    raw = np.nan_to_num(raw, nan=0.0)
    if side == "long":
        raw = np.clip(raw, 0.0, 1.0)
    elif side == "short":
        raw = np.clip(raw, -1.0, 0.0)
    return raw


def member_factors(spec: dict, fields: dict, control: str | None = None, seed: int = 0) -> tuple[list, dict]:
    """Grid members and the factor matrix of each distinct expression.

    `control` is only set by calibration: "random" replaces factors with noise,
    "shuffle" block-permutes them in time, "leak" uses the future return itself.
    """
    combos = specs.expand(spec)
    expressions = sorted({c["expression"] for c in combos})
    keys = {e: f"e{i}" for i, e in enumerate(expressions)}
    values = evaluate({keys[e]: e for e in expressions}, fields)
    if control:
        rng = np.random.default_rng(seed)
        close = fields["close"]
        for key, value in values.items():
            if control == "random":
                values[key] = rng.normal(size=value.shape)
            elif control == "shuffle":
                block = 720
                starts = np.arange(0, value.shape[0], block)
                order = np.concatenate([np.arange(s, min(s + block, value.shape[0])) for s in rng.permutation(starts)])
                values[key] = value[order]
            elif control == "leak":
                leak = np.full_like(close, np.nan)
                leak[:-25] = close[25:] / close[1:-24] - 1.0
                values[key] = leak
            else:
                raise ValueError(f"unknown control {control!r}")
    return [dict(c, key=keys[c["expression"]]) for c in combos], values


def member_positions(spec: dict, combos: list, values: dict, direction: int) -> list[np.ndarray]:
    rule, side = spec["signal"]["rule"], spec["signal"]["side"]
    z = {}
    if rule != "sign":
        by_window: dict[int, list[str]] = {}
        for c in combos:
            by_window.setdefault(int(c["signal_params"]["window"]), []).append(c["key"])
        for window, keys in by_window.items():
            keys = sorted(set(keys))
            for key, matrix in zip(keys, zscore([values[k] for k in keys], window)):
                z[(key, window)] = matrix
    return [
        position(
            values[c["key"]],
            None if rule == "sign" else z[(c["key"], int(c["signal_params"]["window"]))],
            rule,
            c["signal_params"],
            direction,
            side,
        )
        for c in combos
    ]


# ---------------------------------------------------------------- backtest


def targets(positions: np.ndarray, step: float, symbols: int) -> np.ndarray:
    """Quantized equal-allocation targets; NaN means "no order this bar".

    A target is only sent when the quantized position changes, so a held position
    drifts instead of rebalancing every bar. Orders computed at bar t execute at t+1.
    """
    weights = np.round(positions / step) * step / symbols
    changed = np.ones(weights.shape, dtype=bool)
    changed[1:] = weights[1:] != weights[:-1]
    orders = np.where(changed, weights, np.nan)
    out = np.full_like(orders, np.nan)
    out[1:] = orders[:-1]
    return out


def backtest(close: np.ndarray, target_blocks: list[np.ndarray], costs: dict, fee_multipliers: list[float]) -> dict:
    """One alphax sweep: each (target block, fee multiplier) is one shared-cash account."""
    symbols = close.shape[1]
    blocks = [(t, m) for t in target_blocks for m in fee_multipliers]
    groups = len(blocks)
    cash = float(costs["init_cash"])
    stacked = np.concatenate([t for t, _ in blocks], axis=1)
    fees = np.concatenate([[costs["fees"] * m] * symbols for _, m in blocks])
    slippage = np.concatenate([[costs["slippage"] * m] * symbols for _, m in blocks])
    pf = alphax.from_orders(
        np.asfortranarray(close),
        np.asfortranarray(stacked),
        size_type="targetpercent",
        init_cash=[cash] * groups,
        fees=fees,
        slippage=slippage,
        ann_factor=BARS_PER_YEAR,
        threads=int(costs.get("threads", 4)),
        cash_sharing=True,
        group_lens=[symbols] * groups,
        call_seq="auto",
        column_map=np.tile(np.arange(symbols), groups),
    )
    values = np.column_stack([pf.value(group=g) for g in range(groups)])
    orders = pf.orders()
    group = np.asarray(orders["col"], dtype=np.int64) // symbols
    notional = np.abs(np.asarray(orders["size"], dtype=float) * np.asarray(orders["price"], dtype=float))
    return {
        "values": values,
        "fees": np.bincount(group, weights=np.asarray(orders["fees"], dtype=float), minlength=groups),
        "turnover": np.bincount(group, weights=notional, minlength=groups) / cash,
        "orders": np.bincount(group, minlength=groups),
    }


def daily_returns(values: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    """Returns between the last bars of consecutive UTC days (hourly returns are autocorrelated)."""
    day = time_ns // DAY_NS
    last = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
    closes = np.r_[values[0], values[last]]
    return closes[1:] / closes[:-1] - 1.0


def stats(values: np.ndarray, time_ns: np.ndarray) -> dict:
    hourly = values[1:] / values[:-1] - 1.0
    daily = daily_returns(values, time_ns)
    total = float(values[-1] / values[0] - 1.0)
    years = (time_ns[-1] - time_ns[0]) / (365.0 * DAY_NS)
    std = daily.std(ddof=1) if daily.size > 1 else 0.0
    peak = np.maximum.accumulate(values)
    return {
        "sharpe": float(hourly.mean() / hourly.std() * math.sqrt(BARS_PER_YEAR)) if hourly.std() > 0 else 0.0,
        "t_stat": float(daily.mean() / std * math.sqrt(daily.size)) if std > 0 else 0.0,
        "days": int(daily.size),
        "total_return": total,
        "annualized_return": float((1 + total) ** (1 / years) - 1) if total > -1 and years > 0 else -1.0,
        "max_drawdown": float((values / peak - 1.0).min()),
    }


def by_year(values: np.ndarray, time_ns: np.ndarray) -> dict:
    returns = values[1:] / values[:-1] - 1.0
    years = time_ns[1:].astype("datetime64[ns]").astype("datetime64[Y]").astype(np.int64) + 1970
    out = {}
    for year in np.unique(years):
        r = returns[years == year]
        out[str(int(year))] = float(r.mean() / r.std() * math.sqrt(BARS_PER_YEAR)) if r.std() > 0 else 0.0
    return out


# ---------------------------------------------------------------- IC (reference only)


def labels(close: np.ndarray, horizon: int) -> np.ndarray:
    """Return from the next bar's close (execution) over `horizon` bars."""
    out = np.full_like(close, np.nan)
    n = close.shape[0] - 1 - horizon
    if n > 0:
        out[:n] = close[1 + horizon :] / close[1 : 1 + n] - 1.0
    return out


def _rank_z(x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(x)).astype(float)
    return (ranks - ranks.mean()) / ranks.std() if ranks.std() > 0 else np.full_like(ranks, np.nan)


def ic_table(factor: np.ndarray, label: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monthly contributions to the whole-window Spearman IC: (months, symbols)."""
    unique = np.unique(months)
    table = np.full((unique.size, factor.shape[1]), np.nan)
    for s in range(factor.shape[1]):
        ok = np.isfinite(factor[:, s]) & np.isfinite(label[:, s])
        if ok.sum() < MIN_IC_SAMPLES:
            continue
        product = _rank_z(factor[ok, s]) * _rank_z(label[ok, s])
        month = months[ok]
        for i, m in enumerate(unique):
            rows = month == m
            if rows.sum() >= MIN_IC_SAMPLES:
                table[i, s] = product[rows].mean()
    return unique, table


def ic_summary(table: np.ndarray) -> dict:
    with np.errstate(invalid="ignore"):
        pooled = np.nanmean(table, axis=1) if table.size else np.array([])
    series = pooled[np.isfinite(pooled)]
    mean = float(series.mean()) if series.size else float("nan")
    std = float(series.std(ddof=1)) if series.size > 1 else float("nan")
    return {"mean_ic": mean, "icir": mean / std if np.isfinite(std) and std > 0 else float("nan"), "months": int(series.size)}


def months_of(time_ns: np.ndarray) -> np.ndarray:
    return time_ns.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64)


def ic_with_placebo(z: np.ndarray, close: np.ndarray, time_ns: np.ndarray, horizon: int, placebos: int, seed: int) -> dict:
    """IC of a causally z-scored factor plus a p-value from circularly shifted labels."""
    label = labels(close, horizon)
    months = months_of(time_ns)
    out = ic_summary(ic_table(z, label, months)[1])
    if placebos and np.isfinite(out["mean_ic"]):
        rng = np.random.default_rng(seed + horizon)
        n = label.shape[0]
        null = np.array(
            [
                ic_summary(ic_table(z, np.roll(label, int(rng.integers(n // 10, n - n // 10)), axis=0), months)[1])["mean_ic"]
                for _ in range(placebos)
            ]
        )
        null = null[np.isfinite(null)]
        out["placebo_p_value"] = float((1 + np.sum(np.abs(null) >= abs(out["mean_ic"]))) / (1 + null.size))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def alpha158_max_corr(z: np.ndarray, fields: dict, window: int, cache_dir: str | None, cache_key: str) -> dict:
    """Largest |corr| between the z-scored factor and z-scored Alpha158 (reference only)."""
    cache = Path(cache_dir).expanduser() / f"alpha158_z_{cache_key}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        with np.load(cache) as stored:
            reference = {name: stored[name] for name in stored.files}
    else:
        alpha = alphax.alpha158()
        values = alpha.evaluate({name: np.asfortranarray(fields[name]) for name in alpha.required_fields})
        names = list(values)
        reference = {}
        for start in range(0, len(names), 16):
            chunk = names[start : start + 16]
            for name, matrix in zip(chunk, zscore([np.asarray(values[n], dtype=float) for n in chunk], window)):
                reference[name] = matrix.ravel().astype(np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **reference)
    flat = z.ravel()
    corr = {name: _corr(flat, ref.astype(float)) for name, ref in reference.items()}
    corr = {k: v for k, v in corr.items() if np.isfinite(v)}
    closest = max(corr, key=lambda k: abs(corr[k]), default=None)
    return {"max_abs_corr": abs(corr[closest]) if closest else 0.0, "closest": closest}


# ---------------------------------------------------------------- composite and evaluation


def _selection_scores(returns: np.ndarray, metric: str) -> np.ndarray:
    if metric == "mean_return":
        return returns.mean(axis=0)
    std = returns.std(axis=0)
    return np.where(std > 0, returns.mean(axis=0) / np.where(std > 0, std, 1.0), -np.inf)


def composite(spec: dict, positions: list[np.ndarray], close: np.ndarray, costs: dict, step: float) -> tuple[np.ndarray, dict]:
    """The pre-registered use of the grid: an equal-weight ensemble, or a causal walk-forward pick."""
    selection = spec["selection"]
    if selection["method"] == "ensemble":
        return np.mean(positions, axis=0), {"method": "ensemble", "members": len(positions)}
    symbols = close.shape[1]
    result = backtest(close, [targets(p, step, symbols) for p in positions], costs, [1.0])
    member_returns = result["values"][1:] / result["values"][:-1] - 1.0  # row t-1 -> return realised at bar t
    lookback, refit = selection["lookback_bars"], selection["refit_bars"]
    out = np.zeros_like(positions[0])
    picks = []
    for start in range(lookback, close.shape[0], refit):
        window = member_returns[start - lookback : start]  # returns realised up to bar `start`'s close
        pick = int(np.argmax(_selection_scores(window, selection["metric"])))
        out[start : start + refit] = positions[pick][start : start + refit]
        picks.append(pick)
    counts = np.bincount(picks, minlength=len(positions)).tolist() if picks else []
    return out, {"method": "walk_forward", "refits": len(picks), "pick_counts": counts, "flat_warmup_bars": lookback}


def _window_rows(time_ns: np.ndarray, start: str, end: str | None) -> np.ndarray:
    rows = time_ns >= bars.time_ns(start)
    if end:
        rows &= time_ns < bars.time_ns(end)
    return rows


def evaluate_window(
    spec: dict,
    card: dict,
    panel: dict,
    window: tuple[str, str | None],
    config: dict,
    *,
    robustness: bool,
    control: str | None = None,
) -> dict:
    """Evaluates the composite position on the rows of `window`; earlier rows of `panel` are warm-up."""
    fields = {name: panel[name] for name in bars.FIELDS}
    close = panel["close"]
    time_ns = panel["time"]
    direction = int(card["direction"])
    step = float(config["position_step"])
    costs = config["costs"]
    combos, values = member_factors(spec, fields, control, config.get("seed", 0))
    positions = member_positions(spec, combos, values, direction)
    comp, selection = composite(spec, positions, close, costs, step)

    rows = _window_rows(time_ns, *window)
    if rows.sum() < 24 * 60:
        raise ValueError(f"window {window} has only {int(rows.sum())} bars")
    t_rows = time_ns[rows]
    symbols = close.shape[1]
    main = backtest(close[rows], [targets(comp[rows], step, symbols)], costs, [0.0, 1.0, 2.0])
    by_cost = {f"{m:g}x": stats(main["values"][:, i], t_rows) for i, m in enumerate((0.0, 1.0, 2.0))}
    out = {
        "window": {"start": window[0], "end": window[1], "bars": int(rows.sum()), "symbols": panel["symbols"]},
        "selection": selection,
        "net": by_cost["1x"] | {
            "orders": int(main["orders"][1]),
            "turnover": float(main["turnover"][1]),
            "fees": float(main["fees"][1]),
        },
        "cost_sensitivity": {k: v["sharpe"] for k, v in by_cost.items()},
        "sharpe_2x": by_cost["2x"]["sharpe"],
        "by_year_sharpe": by_year(main["values"][:, 1], t_rows),
        "daily_returns_1x": daily_returns(main["values"][:, 1], t_rows).tolist(),
        "grid_members": len(combos),
    }
    if not robustness:
        return out
    members = backtest(close[rows], [targets(p[rows], step, symbols) for p in positions], costs, [1.0])
    member_sharpe = [stats(members["values"][:, i], t_rows)["sharpe"] for i in range(len(combos))]
    out["members"] = {
        "sharpe": member_sharpe,
        "positive_share": float(np.mean(np.array(member_sharpe) > 0)),
        "params": [{**c["factor_params"], **c["signal_params"]} for c in combos],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN warm-up rows
        z = np.nanmean(zscore([values[k] for k in sorted({c["key"] for c in combos})], config["normalize_window"]), axis=0)
    out["ic"] = {
        str(h): ic_with_placebo(z[rows], close[rows], t_rows, h, config["placebos"] if h == spec["horizon"] else 0, config.get("seed", 0))
        for h in config["horizons"]
    }
    if config.get("alpha158_reference") and not control:
        key = hashlib.sha256(json.dumps([panel["symbols"], window, int(t_rows[0]), int(t_rows[-1])]).encode()).hexdigest()[:16]
        out["alpha158"] = alpha158_max_corr(
            z[rows], {name: fields[name][rows] for name in bars.FIELDS}, config["normalize_window"], config.get("cache_dir"), key
        )
    return out


def contract_check(spec: dict, symbols: int, seed: int, step: float, costs: dict) -> str | None:
    """Synthetic-bar check: compiles, no future dependency, usable values, prefix-consistent. None if fine."""
    panel = bars.synthetic(seed, rows=800, cols=symbols)
    fields = {name: panel[name] for name in bars.FIELDS}
    cut = 600
    prefix = {name: np.asfortranarray(value[:cut]) for name, value in fields.items()}
    try:
        combos = specs.expand(spec)
        expressions = {f"e{i}": e for i, e in enumerate(sorted({c["expression"] for c in combos}))}
        values = evaluate(expressions, fields)
        early = evaluate(expressions, prefix)
        for key, expression in expressions.items():
            tail = values[key][200:]
            if np.isfinite(tail).mean() < 0.5:
                return f"{expression}: more than half of the values are NaN or infinite"
            if np.nanstd(tail) == 0:
                return f"{expression}: the factor is constant"
            if not np.allclose(early[key], values[key][:cut], rtol=1e-9, atol=1e-12, equal_nan=True):
                return f"{expression}: earlier values change when later bars are appended"
        combos, values = member_factors(spec, fields)
        positions = member_positions(spec, combos, values, 1)
        comp, _ = composite(spec, positions, panel["close"], costs, step)
        backtest(panel["close"], [targets(comp, step, symbols)], costs, [1.0])
    except (specs.SpecError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return None
