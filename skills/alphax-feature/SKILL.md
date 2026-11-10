---
name: alphax-feature
description: Add or change a feature in the alphax engine (Rust crates and the Python binding) without breaking its invariants — where code goes, the constraints every change must keep (bit-identical defaults, records as the only source, thread invariance, the 3% instruction budget), which tests and checks to run and when, and how to update the knowledge base. Use when implementing anything under crates/ or crates/alphax-py.
license: Apache-2.0 (the repository licence)
compatibility: Rust stable with clippy, Python via uv, valgrind for the instruction benchmark.
metadata:
  author: alphax maintainers
  version: "1.0"
  canonical-path: skills/alphax-feature
---

# Adding a feature to alphax

Read `AGENTS.md`. The knowledge base (nivalume/alphax-docs) explains how to
read each crate.

## 1. Where code goes

| Kind of change | Crate |
|---|---|
| One order's arithmetic (price, size, fees, cash) | `alphax-exec` (the kernel) |
| The bar loop: signals, stops, limit orders, cash flows, margin | `alphax-sim` (`from_orders.rs`, `from_signals.rs`) |
| Records and their bytes | `alphax-records` |
| Anything computed after a run | `alphax-analysis` (from records only) |
| Factor expressions | `alphax-expr` (`parse.rs` arity, `graph.rs` node, `eval.rs` kernel) |
| Indicators | `alphax-indicators` (batch and `stream.rs`) |
| Python | `crates/alphax-py/src/*.rs` binding, `python/alphax/*.py` wrapper |

Library crates never download or write files; network code lives in
`examples/`.

## 2. Invariants

- **Defaults unchanged.** With the new parameter at its default, every output
  is bit-identical to before. `tests/fixtures/*.tsv` are golden
  outputs: never regenerate one to make a change pass.
- **Records are the only source.** Anything that moves cash or a position is
  a record; analysis rebuilds balances from records.
- **Threads do not change results.** Serial and parallel runs are
  bit-identical; random draws are functions of `(seed, row, col)`. Test 1, 3
  and N threads.
- **Floats** are compared through `alphax_core::math` (`is_close`, `add`);
  `clippy::float_cmp` is denied.
- **Cost when off ≤ 3%** in instructions:
  `python3 benchmarks/instructions.py --base <commit before the change>`.
  Put new branches in cold helpers; a feature that is off should cost one
  predictable branch.
- **Continuing runs.** If the feature keeps state across bars, it must be in
  `alphax_sim::snapshot::SimState`, and `crates/alphax-sim/tests/resume.rs`
  must still split anywhere bit for bit.

## 3. Tests, by what the feature is

| Source of truth | Test |
|---|---|
| A golden fixture covers it | a case in `tests/fixtures/` and its replay (see skill `alphax-testing`) |
| New behaviour | hand-computed scenarios plus property tests: degenerate default, conservation (cash and value from records), linearity, split invariance, thread invariance |
| Any | a mutation check: remove the key line, the test must fail; then restore |

## 4. Order of work

Write the whole feature first; run the checks at the end of the module, not
after every few lines:

```bash
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv sync --reinstall-package alphax && uv run --with pytest python -m pytest crates/alphax-py/tests -q
python3 benchmarks/instructions.py --base <base>      # only if alphax-exec or alphax-sim changed
```

## 5. Finish

- Commit with a message that says what changed and why; push to the working
  branch.
- Update the knowledge base pages that describe the changed code, pinned to
  the new commit with line numbers checked (each page names its commit in the "行号对应" row of its source table; re-check every line reference when re-pinning).
