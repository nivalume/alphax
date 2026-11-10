# AGENTS.md

Instructions for coding agents (Codex, Grok Build, Claude Code via `CLAUDE.md`, and any tool that reads this file).

## What this repository is

alphax: a parallel, factor-driven vectorized backtesting engine in Rust with a Python package.
Read `README.md`, then the source knowledge base (https://alphax-docs.vercel.app, repository
nivalume/alphax-docs: how to read the code, Chinese) before changing anything non-trivial.

## Commands

```bash
cargo test --workspace                                   # all Rust tests; replays committed golden fixtures, no Python needed
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
uv sync && uv run --with pytest python -m pytest crates/alphax-py/tests -q   # Python package
```

## Rules that are easy to break

- `tests/fixtures/*.tsv` are golden expected outputs; never regenerate them to make a change pass.
- Financial comparisons go through `alphax_core::math` (`is_close`, `add`), never `==` / `<=` on floats; `clippy::float_cmp` is denied.
- Serial and threaded runs must stay bit-identical; random draws are functions of `(seed, row, col)`.
- Library crates never download or write files; network code lives in `examples/`.
- Analysis rebuilds balances from records only: anything that changes cash or a position must be a record.
- The knowledge base (nivalume/alphax-docs) cites code as links to one pinned commit of this repository, with
  line numbers checked against it; when code a page describes changes, update the page there (each page names its commit in the "行号对应" row of its source table; re-check every line reference when re-pinning).

## Skills

Reusable skills live in `skills/` (Agent Skills format, one `SKILL.md` per directory) and are linked into
`.claude/skills/`, `.agents/skills/` and `.grok/skills/` so every tool discovers them.

| Skill | When to use |
|---|---|
| `skills/docs-canvas` | Docs overview / architecture walkthrough / API reference as a navigable Cursor Canvas; requires Cursor Canvas SDK. |
| `skills/mermaid-diagrams` | Any Markdown with a ```mermaid block: pick the diagram type, follow the house style, then run `python3 skills/mermaid-diagrams/scripts/check_mermaid.py <file-or-dir>` and fix every `FAIL` before committing. |
| `skills/alphax-strategy` | Backtest an idea with the Python API: signals, runs, sweeps, and the checks before reporting a number. |
| `skills/alphax-feature` | Implement anything under `crates/`: where code goes, the invariants, which checks to run and when, knowledge-base updates. |
| `skills/alphax-testing` | A golden test fails or a feature needs coverage: replay, diagnose, extend fixtures; property and mutation checks. |

Codex: `$docs-canvas`, `$mermaid-diagrams`, `$alphax-strategy`, ... or `/skills`. Grok Build: `grok inspect` lists what was discovered. Claude Code: loads it automatically.
