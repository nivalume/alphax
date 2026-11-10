---
name: alphax-testing
description: Run and extend alphax's golden-fixture tests — committed expected outputs for the kernel, loops, analysis, signals, arrays and indicators, the Qlib Alpha158/360 references, the WorldQuant 101 interpreter — and the property and mutation checks used where there is no golden file. Use when a change touches numbers a fixture pins, when adding coverage for a feature, or when a golden test fails and you need to tell a real behaviour change from a fixture problem.
license: Apache-2.0 (the repository licence)
compatibility: cargo test replays committed fixtures with no Python. Regenerating the Qlib fixtures needs a Qlib conda env; the WQ101 generator needs only NumPy.
metadata:
  author: alphax maintainers
  version: "1.0"
  canonical-path: skills/alphax-testing
---

# Golden tests

Full detail is in `tests/generators/README.md`; this is the working order.

## 1. Replaying is free

`cargo test --workspace` replays every committed fixture; nothing to install.
The fixtures in `tests/fixtures/` are expected outputs of the engine's
documented behaviour. A golden failure means the engine's output changed.
Find out why before touching anything:

1. `git log -- tests/fixtures/<file>`: did the fixture change? It should not
   have.
2. Run the one failing case and print both sides; most fixture rows name the
   case.
3. Decide: a bug in the change (fix the code), a deliberate change of
   behaviour (say so in the test's module docs and in the commit message, and
   re-derive the expected value by hand before touching the fixture), or a
   fixture row that was never right (keep the rest, skip the row with a
   comment saying why).

Never regenerate a fixture to make a test pass.

## 2. What pins what

| Area | Fixture | Test |
|---|---|---|
| Kernel, sequences, loops, analysis, signals, arrays, indicators | `tests/fixtures/*.tsv` | `crates/*/tests/golden*.rs` |
| Alpha158 / Alpha360 | `qlib_alpha158.tsv`, `qlib_alpha360.tsv` (generated with Qlib, MIT; the expressions are Qlib's) | `parity_qlib.rs` in `alphax-expr`'s tests |
| WorldQuant 101 | `wq101.tsv` (`~` marks tie-dependent cells) from `tests/generators/generate_wq101.py fixture` (NumPy only) | `alphax-expr/tests/wq101.rs` |
| Margin accounts | `margin.tsv` | `alphax-analysis/tests/golden_margin.rs` |
| Shared accounts, Python boundary | `shared_accounts.json` | `crates/alphax-py/tests/test_shared_accounts.py` |

The three generators in `tests/generators/` rebuild the Qlib and WQ101
fixtures. Regenerate only when changing what a case covers or the Qlib
version, then read the diff line by line: a changed value is a finding.

## 3. Adding coverage

Add a case that is awkward on purpose (gaps, a constant series, a short
window, huge and tiny magnitudes) with its expected output worked out by hand
or taken from an independent calculation, and add the replay. A fixture of
well-behaved numbers hides exactly the differences worth finding.

## 4. Without a golden file

Hand-compute a small scenario, then property tests: default parameters
bit-identical, cash and value rebuilt from records, `multiplier = k` equals
size × k, a run split anywhere (`resume`) equals one run, 1/3/N threads
identical. Then a mutation check: delete the line the feature hinges on and
confirm a test fails.
