"""Command line interface: ``alphax <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from alphax.app import AlphaXApplication
from alphax.config.settings import AlphaXSettings
from alphax.core.registry import scenario_registry


class AlphaXCommandLine:
    def __init__(self) -> None:
        self._parser = self._build_parser()

    def main(self, argv: Sequence[str] | None = None) -> int:
        args = self._parser.parse_args(argv)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        for noisy in ("httpx", "LiteLLM", "litellm"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        return int(args.handler(args) or 0)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="alphax", description="LLM-driven alpha research")
        parser.add_argument("-v", "--verbose", action="store_true")
        commands = parser.add_subparsers(dest="command", required=True)

        scenarios = commands.add_parser("scenarios", help="list the available scenarios")
        scenarios.set_defaults(handler=self._scenarios)

        prepare = commands.add_parser("prepare", help="check the environment and export the scenario data")
        prepare.add_argument("-c", "--config", type=Path, help="YAML run file")
        prepare.add_argument("--force", action="store_true", help="export the data again")
        prepare.set_defaults(handler=self._prepare)

        run = commands.add_parser("run", help="run research loops")
        run.add_argument("-c", "--config", type=Path, help="YAML run file")
        run.add_argument("--loops", type=int, help="number of loops (default: workflow.max_loops)")
        run.add_argument("--resume", type=Path, help="continue the run in this directory")
        run.set_defaults(handler=self._run)

        report = commands.add_parser("report", help="print the summary of a finished run")
        report.add_argument("run_dir", type=Path)
        report.set_defaults(handler=self._report)
        return parser

    @staticmethod
    def _scenarios(args: argparse.Namespace) -> int:
        for name in scenario_registry.names():
            scenario = scenario_registry.get(name)
            print(f"{name:<20} {scenario.title}")
        return 0

    @staticmethod
    def _prepare(args: argparse.Namespace) -> int:
        AlphaXApplication(AlphaXSettings.load(args.config)).prepare(force=args.force)
        print("Scenario prepared.")
        return 0

    @staticmethod
    def _run(args: argparse.Namespace) -> int:
        settings = AlphaXSettings.load(args.config)
        report = AlphaXApplication(settings).run(loops=args.loops, resume=args.resume)
        print(report.render())
        return 0

    @staticmethod
    def _report(args: argparse.Namespace) -> int:
        summary_path = args.run_dir / "summary.json"
        if not summary_path.exists():
            print(f"No summary.json in {args.run_dir}", file=sys.stderr)
            return 1
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        headline = summary.get("headline_metrics") or []

        def metrics(values: dict | None) -> str:
            if not values:
                return "(no result)"
            return ", ".join(f"{name}={values[name]:.4f}" for name in headline if name in values) or "-"

        print(f"baseline  {metrics(summary.get('baseline'))}")
        for loop in summary["loops"]:
            if loop["failed_stage"]:
                status = f"failed at {loop['failed_stage']}"
            else:
                status = "PROMOTED" if loop["decision"] else "not promoted"
            print(f"loop {loop['loop']:3d}  {status:<20} {metrics(loop['metrics'])}  {loop['hypothesis']}")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    return AlphaXCommandLine().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
