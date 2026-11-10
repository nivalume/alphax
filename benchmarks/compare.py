#!/usr/bin/env python3
"""Summarize benchmark results and gate on a committed baseline.

Two jobs:

**Summary.** Prints the engine's throughput per workload and thread count. The
checksums are a hard gate: rows for one workload must agree on the checksum and
the fill count (a serial and a threaded run did the same work), or the script
aborts rather than printing a number. A "faster" number that turns out to be
less work is worse than no number.

**The baseline.** ``--check-baseline`` compares the current run against the
committed ``baseline.tsv`` and fails on a regression beyond a threshold. This
is for catching a real regression, not for micro-tuning: these timings come
from a shared VM. Best-of-7 sampling puts the single-threaded noise floor at
about 1.3%, so the 10% default threshold has room; threaded rows are far
noisier and are excluded by default.

Usage::

    python benchmarks/compare.py                       # summary
    python benchmarks/compare.py --check-baseline      # regression gate
    python benchmarks/compare.py --write-baseline      # accept current as the baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks/results"
BASELINE = REPO_ROOT / "benchmarks/baseline.tsv"


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["workload"], row["impl"], row["threads"])


def verify_same_work(rows: list[dict[str, str]]) -> None:
    """Aborts if rows of one workload disagree about the input they generated.

    Serial and threaded runs must produce the same records from the same
    inputs. Nothing but this check stands between a subtle divergence and a
    comparison of two different workloads.
    """
    by_workload: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        entry = by_workload.setdefault(row["workload"], {})
        entry.setdefault(row["checksum"], set()).add(row["impl"])

    problems = []
    for workload, sums in by_workload.items():
        if len(sums) > 1:
            detail = "; ".join(f"{c} <- {sorted(impls)}" for c, impls in sums.items())
            problems.append(f"  {workload}: {detail}")
        fills = {r["fills"] for r in rows if r["workload"] == workload}
        if len(fills) > 1:
            problems.append(f"  {workload}: fill counts differ: {sorted(fills)}")

    if problems:
        print("ABORT: the rows did not do the same work.\n", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print(
            "\nThe input generator in bench.rs has diverged between runs, "
            "or a simulation produced different fills. Fix that before reading any timing.",
            file=sys.stderr,
        )
        sys.exit(2)


def fmt_rate(v: float) -> str:
    return f"{v / 1e6:.1f}M"


def summary(rows: list[dict[str, str]]) -> None:
    seen: list[str] = []
    for r in rows:
        if r["workload"] not in seen:
            seen.append(r["workload"])
    threads = sorted({int(r["threads"]) for r in rows})
    rate = {key(r): float(r["elems_per_sec"]) for r in rows}

    print("\nKERNEL: the simulation loop alone (millions of elements/sec, best of N)\n")
    header = f"{'workload':<22}" + "".join(f"{f'{t} thread' + ('s' if t > 1 else ''):>14}" for t in threads)
    print(header)
    print("-" * len(header))
    for w in seen:
        line = f"{w:<22}"
        for t in threads:
            v = rate.get((w, "alphax", str(t)))
            line += f"{fmt_rate(v) if v else '-':>14}"
        print(line)


def looks_uniform(ratios: dict[tuple[str, str, str], float], faster: bool) -> bool:
    """Whether a set of changes looks like the machine, not the code.

    A code change is selective: it moves the workloads that exercise the
    changed path and leaves the others alone. A machine that is busier — or
    quieter — than the one the baseline came from scales everything by roughly
    one factor. So a change that lands on *every* workload in the same
    direction is evidence about the machine, and saying so is the difference
    between a gate people act on and a gate people learn to ignore.

    Both directions have now been seen in practice, and the second was the
    more dangerous one:

    - An unchanged build reported -40% to -43% across all ten workloads at a
      load average of 1.37 on four cores.
    - An unchanged hot path reported **+35% to +55%** across all ten, because
      the baseline had been taken minutes after a package install while the
      machine was still busy. A uniform improvement is the same signal, and
      it is worse, because nobody investigates good news — they refresh the
      baseline and bake the busy machine's numbers into the next comparison.

    A baseline is only as good as the quiet it was measured in.
    """
    values = list(ratios.values())
    if len(values) < 4:
        return False
    moved = [v for v in values if (v > 1.0 if faster else v < 1.0)]
    if len(moved) < len(values):
        return False
    # Every row moved the same way, within a band of each other. The band is
    # wider than the 15% used for slowdowns because a quiet machine helps the
    # memory-bound workloads more than the compute-bound ones.
    return (max(values) - min(values)) / max(values) < 0.40


def check_baseline(current: list[dict[str, str]], tolerance: float, include_threaded: bool) -> int:
    base = {key(r): float(r["elems_per_sec"]) for r in read(BASELINE)}
    if not base:
        print(f"no baseline at {BASELINE}; run with --write-baseline first", file=sys.stderr)
        return 1

    # Threaded rows are excluded by default, and the reason is measured rather
    # than assumed. Running the suite four times unchanged on the reference
    # machine gave this spread between runs:
    #
    #     1 thread    median 1.3%   worst  3.7%
    #     2 threads   median 4.1%   worst 21.9%
    #     4 threads   median 5.9%   worst 47.3%
    #
    # The single-threaded figures are with best-of-7 sampling; at best-of-3
    # they were 3.4% / 9.5%, wide enough to hide a 10% regression.
    #
    # A gate whose noise floor is wider than the regressions it is meant to
    # catch does not gate anything — it just teaches you to ignore it. The
    # threaded numbers stay in the results for the summary; they
    # are not evidence of a regression on a shared machine.
    if not include_threaded:
        current = [r for r in current if r["threads"] == "1"]

    regressions, improvements, missing = [], [], []
    ratios: dict[tuple[str, str, str], float] = {}
    for row in current:
        k = key(row)
        if k not in base:
            missing.append(k)
            continue
        now, before = float(row["elems_per_sec"]), base[k]
        ratios[k] = now / before
        change = now / before - 1.0
        label = f"{k[0]} ({k[1]}, {k[2]}t)"
        if change < -tolerance:
            regressions.append(f"  {label:<40} {before / 1e6:.1f}M -> {now / 1e6:.1f}M  ({change:+.0%})")
        elif change > tolerance:
            improvements.append(f"  {label:<40} {before / 1e6:.1f}M -> {now / 1e6:.1f}M  ({change:+.0%})")

    if improvements:
        print(f"\nFaster than baseline by more than {tolerance:.0%}:")
        print("\n".join(improvements))
        if looks_uniform(ratios, faster=True):
            print(
                "\nBUT: every workload got faster by close to the same factor "
                f"({min(ratios.values()):.0%} to {max(ratios.values()):.0%} of baseline).\n"
                "That is the signature of a quieter machine, not a faster engine —\n"
                "a code change moves some workloads and not others. Most likely the\n"
                "*baseline* was taken on a busy machine. Check what the diff actually\n"
                "touched before refreshing, or the busy numbers get baked in and the\n"
                "next honest run looks like a regression."
            )
        else:
            print("  (if these are real, refresh the baseline with --write-baseline)")
    if missing:
        print(f"\nNot in the baseline: {len(missing)} entries — refresh it if workloads changed")
    if regressions:
        print(f"\nREGRESSION: slower than baseline by more than {tolerance:.0%}:", file=sys.stderr)
        print("\n".join(regressions), file=sys.stderr)
        if looks_uniform(ratios, faster=False):
            print(
                "\nBUT: every workload regressed by close to the same factor "
                f"({min(ratios.values()):.0%} to {max(ratios.values()):.0%} of baseline).\n"
                "A code regression hits some workloads and not others; a loaded or\n"
                "throttled machine hits all of them by one factor. Check the load\n"
                "average and re-run before reading this as a regression.",
                file=sys.stderr,
            )
        print(
            "\nSingle-threaded timings move about 1.3% between runs on the reference\n"
            "machine, so a result near the threshold deserves a re-run before it\n"
            "deserves an investigation.",
            file=sys.stderr,
        )
        return 1

    scope = "all rows" if include_threaded else "single-threaded rows"
    print(f"\nNo regression beyond {tolerance:.0%} against {BASELINE.name} ({scope}).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-baseline", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="fractional slowdown that counts as a regression (default 0.10)",
    )
    parser.add_argument(
        "--include-threaded",
        action="store_true",
        help="also gate on threaded rows; only useful on a machine quiet enough "
        "for them (see the note in check_baseline)",
    )
    args = parser.parse_args()

    ours = read(RESULTS / "alphax.tsv")
    if not ours:
        print(
            "no results at benchmarks/results/alphax.tsv; run:\n"
            "  cargo run --release --example bench -p alphax-sim",
            file=sys.stderr,
        )
        return 1

    verify_same_work(ours)

    if args.write_baseline:
        header = (
            "# Committed performance baseline for this engine.\n"
            "# Regenerate with: cargo run --release --example bench -p alphax-sim\n"
            "#                  python benchmarks/compare.py --write-baseline\n"
            "# Absolute numbers are machine-specific; the point is catching large\n"
            "# regressions on the same machine, not comparing across machines.\n"
        )
        src = (RESULTS / "alphax.tsv").read_text().splitlines()
        body = [line for line in src if not line.startswith("#")]
        machine = [line for line in src if line.startswith("# cores")]
        BASELINE.write_text(header + "\n".join(machine + body) + "\n")
        print(f"wrote {BASELINE} ({len(body) - 1} entries)")
        return 0

    if args.check_baseline:
        return check_baseline(ours, args.tolerance, args.include_threaded)

    summary(ours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
