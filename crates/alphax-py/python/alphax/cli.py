"""The ``alphax`` command.

::

    alphax run config.toml [--notify-file runs.log] [--notify-command "mail -s alphax me@x"]
    alphax sweep config.toml --set window=10,20,50 [--metric sharpe_ratio]
    alphax schedule config.toml --at 06:30 [--weekdays 1-5]    # prints a crontab line
    alphax help indicators.rsi
    alphax mcp [--source CHECKOUT]                             # the MCP server, over stdio

``run`` and ``sweep`` read a TOML backtest (see ``alphax.config``).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def _notifiers(args: argparse.Namespace, cfg: dict) -> list:
    """The ``[notify]`` section's ``file`` and ``command``, then the
    command line's."""
    from pathlib import Path

    from .schedule import Command, File

    out: list = []
    section = cfg.get("notify", {})
    if "file" in section:
        out.append(File(str(Path(cfg.get("_base", ".")) / section["file"])))
    if "command" in section:
        out.append(Command(section["command"]))
    if args.notify_file:
        out.append(File(args.notify_file))
    if args.notify_command:
        out.append(Command(args.notify_command))
    return out


def _key(stats: dict, metric: str) -> str:
    """The stats key ``metric`` names: exactly, or ignoring case, spaces,
    units and underscores (``sharpe_ratio`` is ``Sharpe Ratio``)."""
    if metric in stats:
        return metric

    def norm(k: str) -> str:
        return "".join(ch for ch in k.split("[")[0].lower() if ch.isalnum())

    for k in stats:
        if norm(k) == norm(metric):
            return k
    raise SystemExit(f"no statistic {metric!r}; there are: {', '.join(stats)}")


def _table(result: dict, metric: str) -> str:
    rows = []
    first = result["stats"][0][0] if result["stats"] and result["stats"][0] else {}
    metric = _key(first, metric) if first else metric
    for params, per_account in zip(result["params"], result["stats"]):
        values = [st.get(metric) for st in per_account]
        label = ", ".join(f"{k}={v}" for k, v in params.items()) or "run"
        rows.append(f"{label:30s} " + " ".join(f"{v if v is not None else 'nan':>12.6}" if isinstance(v, float)
                                               else f"{v!s:>12}" for v in values))
    return f"{metric} per account\n" + "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    from . import config
    from .schedule import Job, crontab
    from .tools import help as alphax_help

    parser = argparse.ArgumentParser(prog="alphax", description="alphax backtests from the command line")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        p = sub.add_parser(name, help=f"{name} a TOML backtest")
        p.add_argument("config")
        p.add_argument("--metric", default="Total Return [%]")
        p.add_argument("--notify-file")
        p.add_argument("--notify-command")
        if name == "sweep":
            p.add_argument("--set", action="append", default=[], metavar="NAME=V1,V2",
                           help="sweep values, replacing [sweep]")
    p = sub.add_parser("schedule", help="print a crontab line that runs a config daily")
    p.add_argument("config")
    p.add_argument("--at", default="06:30")
    p.add_argument("--weekdays", default="*")
    p = sub.add_parser("help", help="a short reference for a alphax object")
    p.add_argument("name")
    p.add_argument("--full", action="store_true")
    p = sub.add_parser("mcp", help="run the MCP server over stdio")
    p.add_argument("--source")
    args = parser.parse_args(argv)

    if args.command in ("run", "sweep"):
        cfg = config.load(args.config)
        if args.command == "sweep":
            for item in args.set:
                name, _, values = item.partition("=")
                cfg.setdefault("sweep", {})[name] = [_value(v) for v in values.split(",")]
        job = Job(lambda: config.run(cfg), name=f"alphax {args.command} {args.config}",
                  notify=_notifiers(args, cfg))
        result = job()
        print(_table(result, args.metric))
    elif args.command == "schedule":
        print(crontab(args.config, at=args.at, weekdays=args.weekdays))
    elif args.command == "mcp":
        from . import mcp

        return mcp.main(["--source", args.source] if args.source else [])
    else:
        alphax_help(args.name, full=args.full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
