"""A backtest described in a TOML file, run without writing Python.

::

    [data]
    path = "bars.npz"            # arrays open, high, low, close, volume, ... and optional time
    # csv = "bars.csv"           # or long rows: time, symbol, open, high, low, close, volume
    # loader = "mypkg.data:load" # or a function returning a dict of arrays

    [signals]                    # factor expressions over the data's fields
    entries = "$close > Mean($close, {window})"
    exits = "$close < Mean($close, {window})"

    [backtest]                   # from_signals keyword arguments
    # warmup = false             # keep signals from partial windows (Qlib's values)
    size = 1.0
    fees = 0.001
    direction = "longonly"

    [sweep]                      # optional: a grid; each combination fills {name} placeholders
    window = [10, 20, 50]

    [output]
    stats = "stats.json"         # statistics per parameter set and account
    portfolio = "run.alphax"     # alphax.save of the last (or only) run

    [notify]                     # read by `alphax run` and `alphax sweep`
    file = "runs.log"            # append each outcome
    command = "mail -s alphax me@example.com"   # or pipe it to a command

Paths are relative to the file. ``entry = "from_orders"`` under
``[backtest]`` takes a ``size`` expression under ``[signals]`` instead.
Signals are evaluated with warm-up (:meth:`alphax.FactorSet.with_warmup`):
no entry, exit or order on a symbol's rows before every window of the
expression has filled; ``warmup = false`` under ``[backtest]`` turns that
off. A ``NaN`` entry or exit is no signal.
``load`` reads and checks a file; ``run`` runs one; the ``alphax`` command
(``alphax run``, ``alphax sweep``) calls them.
"""

from __future__ import annotations

import importlib
import json
import string
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load", "load_data", "run"]

_SECTIONS = {"data", "signals", "backtest", "sweep", "output", "notify"}


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def load(path: str | Path) -> dict:
    """The file's contents, checked, with ``_base`` set to its directory."""
    path = Path(path)
    cfg = _toml(path)
    unknown = set(cfg) - _SECTIONS
    if unknown:
        raise ValueError(f"{path}: unknown section [{sorted(unknown)[0]}]; sections are {sorted(_SECTIONS)}")
    for need in ("data", "signals"):
        if need not in cfg:
            raise ValueError(f"{path}: missing [{need}]")
    entry = cfg.get("backtest", {}).get("entry", "from_signals")
    if entry not in ("from_signals", "from_orders"):
        raise ValueError(f"{path}: entry must be from_signals or from_orders")
    if not isinstance(cfg.get("backtest", {}).get("warmup", True), bool):
        raise ValueError(f"{path}: warmup must be true or false")
    want = ("size",) if entry == "from_orders" else ("entries", "exits")
    missing = [k for k in want if k not in cfg["signals"]]
    if missing:
        raise ValueError(f"{path}: [signals] needs {', '.join(missing)} for {entry}")
    names = {n for s in cfg["signals"].values() for _, n, _, _ in string.Formatter().parse(s) if n}
    absent = names - set(cfg.get("sweep", {}))
    if absent:
        raise ValueError(f"{path}: placeholder {{{sorted(absent)[0]}}} has no values under [sweep]")
    cfg["_base"] = path.parent
    return cfg


def load_data(section: dict, base: Path) -> dict[str, np.ndarray]:
    """The ``[data]`` section's arrays: 2-D ``(bars, symbols)`` per field,
    plus ``time`` when the source has it."""
    if "loader" in section:
        module, _, name = section["loader"].partition(":")
        out = getattr(importlib.import_module(module), name)(**section.get("args", {}))
        return {k: np.asarray(v) for k, v in out.items()}
    if "path" in section:
        with np.load(base / section["path"]) as z:
            return {k: z[k] for k in z.files}
    if "csv" in section:
        return _long_csv(base / section["csv"])
    raise ValueError("[data] needs path, csv or loader")


def _long_csv(path: Path) -> dict[str, np.ndarray]:
    """Long rows ``time, symbol, field...`` pivoted to one matrix per field."""
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = rows.dtype.names
    if "time" not in names or "symbol" not in names:
        raise ValueError(f"{path}: needs time and symbol columns")
    times, t_idx = np.unique(rows["time"].astype(str), return_inverse=True)
    symbols, s_idx = np.unique(rows["symbol"].astype(str), return_inverse=True)
    out: dict[str, np.ndarray] = {}
    for field in names:
        if field in ("time", "symbol"):
            continue
        m = np.full((len(times), len(symbols)), np.nan)
        m[t_idx, s_idx] = rows[field].astype(float)
        out[field] = m
    out["time"] = times.astype("datetime64[ns]")
    out["symbols"] = symbols
    return out


def _combos(cfg: dict) -> list[dict]:
    from .params import grid

    sweep = cfg.get("sweep")
    return grid(**sweep) if sweep else [{}]


def _signal(values: np.ndarray) -> np.ndarray:
    """An entry or exit wherever a factor is non-zero; ``NaN`` (a window
    still warming up, a missing input) is no signal."""
    return (values != 0) & ~np.isnan(values)


def run(cfg: dict | str | Path) -> dict[str, Any]:
    """Runs a configuration: every combination of ``[sweep]`` (or the one
    run), writing what ``[output]`` asks for. Returns ``params``, the
    ``portfolios`` and their ``stats``."""
    import alphax

    if not isinstance(cfg, dict):
        cfg = load(cfg)
    base = Path(cfg.get("_base", "."))
    data = load_data(cfg["data"], base)
    fields = {k: np.asfortranarray(v, dtype=float) for k, v in data.items()
              if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind in "fiu"}
    if "close" not in fields:
        raise ValueError("[data] has no close")
    backtest = dict(cfg.get("backtest", {}))
    entry = backtest.pop("entry", "from_signals")
    warmup = backtest.pop("warmup", True)
    market = {k: fields[k] for k in ("open", "high", "low") if k in fields}
    combos = _combos(cfg)
    portfolios, stats = [], []
    # Every combination in one graph: what no parameter touches is computed once.
    grid = alphax.FactorGrid.compile(cfg["signals"], combos).with_warmup(warmup)
    wide = grid.evaluate(fields)
    n_symbols = fields["close"].shape[1]
    for k, combo in enumerate(combos):
        values = {name: v[:, k * n_symbols:(k + 1) * n_symbols] for name, v in wide.items()}
        if entry == "from_signals":
            pf = alphax.from_signals(fields["close"], _signal(values["entries"]), _signal(values["exits"]),
                                     **market, **backtest)
        else:
            pf = alphax.from_orders(fields["close"], values["size"], **market, **backtest)
        portfolios.append(pf)
        n_accounts = len(pf.group_lens) if pf.cash_sharing else pf.n_cols
        stats.append([_plain(pf.stats(group=a) if pf.cash_sharing else pf.stats(a)) for a in range(n_accounts)])
    out = cfg.get("output", {})
    if "stats" in out:
        with open(base / out["stats"], "w") as f:
            json.dump([{"params": c, "stats": s} for c, s in zip(combos, stats)], f, indent=2)
    if "portfolio" in out:
        alphax.save(portfolios[-1], base / out["portfolio"])
    return {"params": combos, "portfolios": portfolios, "stats": stats}


def _plain(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v if isinstance(v, (str, bool, type(None))) else str(v)
    return out
