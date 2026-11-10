"""Single-asset time-series factor: a alphax expression template whose grid is used as a whole."""

from __future__ import annotations

import numpy as np

from alphax_lab import data as bars
from alphax_lab.protocol.policy import t_required
from alphax_lab.types.factor_ts import harness
from alphax_lab.types.factor_ts import spec as specs


def _check(name: str, passed: bool, value, threshold: str) -> dict:
    return {"name": name, "passed": bool(passed), "value": value, "threshold": threshold}


class FactorTS:
    name = "factor_ts"

    # ------------------------------------------------------------ spec

    def spec_prompt(self) -> str:
        by_arity: dict[int, list[str]] = {}
        for op, arity in specs.ARITY.items():
            by_arity.setdefault(arity, []).append(op)
        return f"""Spec language (alphax, Qlib-style, evaluated per symbol along time on 1h bars):
- Fields: {', '.join('$' + f for f in specs.FIELDS)}. Numbers, + - * / ** and comparisons > >= < <= == != & |.
- One-argument functions: {', '.join(by_arity[1])}. Three-argument: {', '.join(by_arity[3])}.
- Window functions Op(x, n) over the trailing n bars: {', '.join(by_arity[2])}. Rank(x, n) is a time-series
  percentile, not a cross-sectional rank. Ref(x, n) needs n > 0. No cross-symbol operators.
- Windows are in bars: 24 = one day, 168 = one week, 720 = about one month.

Spec JSON:
{{"name": "ma_gap", "expression": "Mean($close,{{fast}})/Mean($close,{{slow}})-1",
  "params": {{"fast": [72, 168, 336], "slow": [720, 1440, 2160]}}, "constraints": ["fast < slow"],
  "horizon": 24,
  "signal": {{"rule": "sign", "side": "both", "params": {{}}}},
  "selection": {{"method": "ensemble"}}}}
- params are an ENSEMBLE: every combination is used and their positions are averaged. Pick values that
  cover the card's param_ranges evenly; every value must lie inside those ranges.
- signal.rule: "sign" (position = sign of the factor; needs a zero-centred factor), "zscore" (params window,
  scale: position = clip(z/scale, -1, 1)), "band" (params window, entry, exit: hold +1 after z > entry until
  z < exit, symmetric for -1). Signal params also need card ranges. side: both | long | short.
- The card's direction is applied by the harness; do not negate the expression yourself.
- selection: "ensemble" by default. Use {{"method": "walk_forward", "metric": "sharpe", "lookback_bars": L,
  "refit_bars": R}} only when the card argues the right parameter changes over time.
- horizon (1, 4 or 24) is only used for the reference IC."""

    def regulate(self, raw_spec: dict, card: dict, tried: dict[str, str], policy: dict) -> tuple[dict | None, list[str]]:
        limits = policy["spec_limits"]
        return specs.regulate(
            raw_spec,
            horizons=policy["harness"]["horizons"],
            max_grid=limits["max_grid"],
            max_nodes=limits["max_nodes"],
            max_free_ratio=limits["max_free_ratio"],
            max_common_subtree=limits["max_common_subtree"],
            others={},
            tried=tried,
            ranges=card["param_ranges"],
        )

    def contract_check(self, spec: dict, card: dict, policy: dict) -> str | None:
        return harness.contract_check(
            spec, len(card["symbols"]), policy["harness"]["seed"], policy["harness"]["position_step"], policy["costs"]
        )

    # ------------------------------------------------------------ runs

    def _config(self, policy: dict) -> dict:
        return {**policy["harness"], "costs": policy["costs"], "cache_dir": "~/.cache/alphax_lab"}

    def _load(self, policy: dict, symbols: list[str], start: str, end: str) -> dict:
        d = policy["data"]
        return bars.load(d["catalog"], symbols, d["venue"], d["interval"], start, end)

    def run(self, spec: dict, card: dict, policy: dict, window: str, *, control: str | None = None) -> dict:
        is_start, is_end = policy["periods"]["in_sample"]
        oos_start, oos_end = policy["periods"]["out_of_sample"]
        config = self._config(policy)
        if window == "in_sample":
            # The only data an in-sample run may load ends at is_end.
            panel = self._load(policy, card["symbols"], is_start, is_end)
            span, load_end = (is_start, is_end), is_end
        elif window == "out_of_sample":
            panel = self._load(policy, card["symbols"], is_start, oos_end)  # in-sample rows are warm-up only
            span, load_end = (oos_start, oos_end), oos_end
        elif window.startswith("forward:"):
            start = window.split(":", 1)[1]
            load_end = str(np.datetime64("now", "D") + 1)
            panel = self._load(policy, card["symbols"], oos_start, load_end)
            span = (start, None)
        else:
            raise ValueError(f"unknown window {window!r}")
        result = harness.evaluate_window(spec, card, panel, span, config, robustness=window == "in_sample", control=control)
        if window == "in_sample":
            cross = {}
            for symbol in policy["data"]["universe"]:
                try:
                    single = self._load(policy, [symbol], is_start, load_end)
                    r = harness.evaluate_window(spec, card, single, span, config, robustness=False, control=control)
                    cross[symbol] = {"sharpe": r["net"]["sharpe"], "t_stat": r["net"]["t_stat"]}
                except (ValueError, OSError) as error:
                    cross[symbol] = {"error": str(error)}
            valid = [v["sharpe"] for v in cross.values() if "sharpe" in v]
            result["cross_asset"] = {
                "by_symbol": cross,
                "positive_share": float(np.mean(np.array(valid) > 0)) if valid else 0.0,
            }
        result["funding_modeled"] = policy["costs"]["funding_modeled"]
        return result

    # ------------------------------------------------------------ gates

    def gate(self, result: dict, policy: dict, tests: int) -> list[dict]:
        g = policy["gate"]
        t, need = result["net"]["t_stat"], t_required(tests, g["alpha"])
        return [
            _check("net_t_stat", t >= need, round(t, 3), f">= {need:.2f} (Bonferroni, N={tests})"),
            _check("not_suspicious", t <= g["suspicious_t"], round(t, 3), f"<= {g['suspicious_t']} (else look-ahead)"),
            _check("sharpe_at_2x_cost", result["sharpe_2x"] > g["min_sharpe_at_2x_cost"], round(result["sharpe_2x"], 3),
                   f"> {g['min_sharpe_at_2x_cost']}"),
            _check("grid_members_positive", result["members"]["positive_share"] >= g["min_member_positive_share"],
                   round(result["members"]["positive_share"], 3), f">= {g['min_member_positive_share']}"),
            _check("cross_asset_positive", result["cross_asset"]["positive_share"] >= g["min_cross_asset_positive_share"],
                   round(result["cross_asset"]["positive_share"], 3), f">= {g['min_cross_asset_positive_share']}"),
        ]

    def oos_gate(self, result: dict, policy: dict, tests: int) -> list[dict]:
        o, g = policy["oos"], policy["gate"]
        t, need = result["net"]["t_stat"], t_required(tests, o["alpha"], two_sided=False)
        return [
            _check("oos_net_t_stat", t >= need, round(t, 3), f">= {need:.2f} (one-sided Bonferroni, M={tests})"),
            _check("not_suspicious", t <= g["suspicious_t"], round(t, 3), f"<= {g['suspicious_t']}"),
            _check("oos_sharpe_at_2x_cost", result["sharpe_2x"] > 0, round(result["sharpe_2x"], 3), "> 0"),
        ]

    # ------------------------------------------------------------ report

    def report_sections(self, result: dict) -> list[tuple[str, str, bool]]:
        net = result["net"]
        sections = [
            (
                "Net performance (1x cost)",
                f"t (daily) {net['t_stat']:.2f} over {net['days']} days; Sharpe {net['sharpe']:.2f}; "
                f"annual return {net['annualized_return']:.1%}; max drawdown {net['max_drawdown']:.1%}; "
                f"{net['orders']} orders; turnover {net['turnover']:.1f}x; fees {net['fees']:.0f}",
                True,
            ),
            ("Cost sensitivity (Sharpe)", ", ".join(f"{k}: {v:.2f}" for k, v in result["cost_sensitivity"].items()), True),
        ]
        if "members" in result:
            m = result["members"]
            rows = "\n".join(f"| {p} | {s:.2f} |" for p, s in zip(m["params"], m["sharpe"]))
            sections.append(
                ("Grid members", f"{m['positive_share']:.0%} positive\n\n| params | Sharpe |\n|---|---|\n{rows}", True)
            )
        if "cross_asset" in result:
            rows = "\n".join(
                f"| {k} | {v.get('sharpe', float('nan')):.2f} | {v.get('t_stat', float('nan')):.2f} |"
                for k, v in result["cross_asset"]["by_symbol"].items()
            )
            sections.append(
                (
                    "Cross-asset (robustness, not extra sample: majors correlate ~0.8)",
                    f"{result['cross_asset']['positive_share']:.0%} positive\n\n| symbol | Sharpe | t |\n|---|---|---|\n{rows}",
                    True,
                )
            )
        sections.append(("Selection", str(result["selection"]), False))
        sections.append(("By year (Sharpe)", ", ".join(f"{k}: {v:.2f}" for k, v in result["by_year_sharpe"].items()), False))
        if "ic" in result:
            ic = "; ".join(
                f"h={h}: IC {v['mean_ic']:.3f}, ICIR {v['icir']:.2f}"
                + (f", placebo p {v['placebo_p_value']:.2f}" if "placebo_p_value" in v else "")
                for h, v in result["ic"].items()
            )
            sections.append(("IC of the z-scored factor", ic, False))
        if "alpha158" in result:
            a = result["alpha158"]
            sections.append(("Closest Alpha158 factor", f"{a['closest']}: |corr| {a['max_abs_corr']:.2f}", False))
        if not result.get("funding_modeled", False):
            sections.append(("Funding", "Perpetual funding payments are NOT included in these returns.", False))
        return sections
