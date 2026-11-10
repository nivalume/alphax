"""alphax-lab command line. The ledger repository is ALPHAX_LAB_LEDGER (default ../alphax-ledger next to alphax)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "alphax-ledger"
LEGACY_MANIFEST = Path(__file__).resolve().parents[1] / "legacy" / "manifest.yaml"


def _lab():
    from alphax_lab.pipeline import Lab

    return Lab(os.environ.get("ALPHAX_LAB_LEDGER", DEFAULT_LEDGER))


def _symbols(text: str) -> list[str]:
    symbols = [s.strip().upper() for s in text.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols is required")
    return symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphax-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="write hypothesis cards + specs to pending/ (the LLM sees no results)")
    p.add_argument("--symbols", required=True)
    p.add_argument("--n", type=int, default=6)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="research direction in your own words")
    source.add_argument("--pdf", help="literature PDF to draw mechanisms from")
    p.add_argument("--pages", help="page range of the PDF to read, e.g. 38-50")

    sub.add_parser("review", help="list pending cards")
    p = sub.add_parser("show", help="print one pending item")
    p.add_argument("id")
    p = sub.add_parser("approve", help="freeze a pending item into the registry and ledger")
    p.add_argument("ids", nargs="+")
    p = sub.add_parser("repair", help="let the LLM fix a spec that failed its contract check")
    p.add_argument("id")
    sub.add_parser("run", help="run every registered spec once on in-sample data")
    p = sub.add_parser("oos", help="run the out-of-sample batch for current in-sample passes")
    p.add_argument("--force", action="store_true", help="run before the batch interval (recorded)")
    sub.add_parser("forward", help="shadow-track out-of-sample survivors on new bars")
    sub.add_parser("report", help="ledger summary")
    p = sub.add_parser("calibrate", help="negative and positive controls")
    p.add_argument("--symbols", default="BTCUSDT")
    p = sub.add_parser("import-legacy", help="count earlier trials on the same data")
    p.add_argument("--manifest", default=str(LEGACY_MANIFEST))
    args = parser.parse_args(argv)

    lab = _lab()
    if args.command == "propose":
        from alphax_lab.llm import LLM

        pages: tuple[int, int] | None = None
        if args.pages:
            page_parts = args.pages.split("-", 1)
            if len(page_parts) != 2:
                parser.error("--pages must be START-END")
            pages = (int(page_parts[0]), int(page_parts[1]))
        items = lab.propose(
            LLM(), symbols=_symbols(args.symbols), n=args.n, prompt=args.prompt, pdf=args.pdf, pages=pages
        )
        for item in items:
            flag = "ok" if not item["problems"] else f"{len(item['problems'])} problem(s)"
            print(f"{item['id']}  {item['card'].get('title', '?')}  [{flag}]")
        print(f"\nReview and edit {lab.root / 'pending'}, then `alphax-lab approve <id>`.")
    elif args.command == "review":
        for item_id, item in lab.pending().items():
            card = item["card"]
            print(f"{item_id}  {card.get('title')}  direction={card.get('direction')}  expr={item['spec'].get('expression')}")
            for problem in item.get("problems") or []:
                print(f"    ! {problem}")
    elif args.command == "show":
        print(yaml.safe_dump(lab.pending()[args.id], sort_keys=False, allow_unicode=True))
    elif args.command == "approve":
        for item_id in args.ids:
            try:
                entry = lab.approve(item_id)
                print(f"registered {entry['spec_id']}  {entry['card']['title']}")
            except ValueError as error:
                print(error, file=sys.stderr)
    elif args.command == "repair":
        from alphax_lab.llm import LLM

        item = lab.repair(LLM(), args.id)
        print(json.dumps(item["spec"], ensure_ascii=False, indent=1))
    elif args.command == "run":
        for entry in lab.run():
            verdict = "error" if "error" in entry else ("PASS" if entry["passed"] else "reject")
            print(f"{entry['spec_id']}  {verdict}  -> {lab.root / 'reports' / (entry['spec_id'] + '.in_sample.md')}")
    elif args.command == "oos":
        entry = lab.oos(force=args.force)
        for sid, result in entry["results"].items():
            print(f"{sid}  {'PASS' if result['passed'] else 'reject'}")
    elif args.command == "forward":
        for row in lab.forward():
            print(f"{row['spec_id']}  {row['status']}  {row.get('days', '')} days")
    elif args.command == "report":
        print(lab.summary())
    elif args.command == "calibrate":
        print(json.dumps(lab.calibrate(_symbols(args.symbols)), indent=1))
    elif args.command == "import-legacy":
        print(f"imported {lab.import_legacy(args.manifest)} legacy entries; N = {lab.ledger.in_sample_tests()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
