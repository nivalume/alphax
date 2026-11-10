---
name: mermaid-diagrams
description: Draw and validate Mermaid diagrams (classDiagram, sequenceDiagram, flowchart, stateDiagram, gantt) inside Markdown so GitHub renders them. Use when writing or editing docs that explain code structure, a call sequence, a decision pipeline, a roadmap or a protocol; when asked for a 类图 / 时序图 / 流程图 / 状态图 / 路线图; and before committing any Markdown that contains a ```mermaid block. Ships a checker that renders every block with mermaid-cli and reports parse errors by file and line.
license: Apache-2.0 (the repository licence)
compatibility: Python 3.9+ for the checker. Rendering needs Node.js with @mermaid-js/mermaid-cli and a Chromium; the checker finds Playwright, Puppeteer and system installs or accepts --chrome.
metadata:
  author: alphax maintainers
  version: "1.0"
  canonical-path: skills/mermaid-diagrams
---

# Mermaid diagrams for code documentation

Diagrams in this repository are Mermaid blocks inside Markdown, because GitHub
renders them, reviewers can diff them, and they can be checked mechanically.
Do not add SVGs by hand.

## 1. Decide whether a diagram earns its place

Draw one when the reader has to hold more than ~5 things in relation at once:
who owns whom, what happens in what order, which branch leads where, what
depends on what. Otherwise a table or a sentence is faster to read and to keep
current. One diagram per idea; two small diagrams beat one large one.

## 2. Pick the type from the question the reader has

| Reader's question | Type | Worked example in this repo |
|---|---|---|
| What are the types and who owns / borrows / implements what? | `classDiagram` | alphax-docs `crates/alphax-sim.md` (类图) |
| What happens, in what order, between which parts, in one unit of work? | `sequenceDiagram` | alphax-docs `algorithms/bar-loop.md` (one bar of a shared account) |
| Which decision leads where; what is the fixed order of stages? | `flowchart TD` | alphax-docs `algorithms/prepare-order.md` (`prepare_order`) |
| What depends on what (crates, tasks, milestones)? | `flowchart BT` / `LR` | alphax-docs `guides/system-map.md` (crate dependencies) |
| What states can a thing be in and what moves it? | `stateDiagram-v2` | alphax-docs `tooling/research-protocol.md` (research protocol) |
| How long will phases take, in what order? | `gantt` | alphax-docs `guides/reading-roadmap.md` |

Templates for each, with the member and arrow syntax that is known to render,
are in [references/diagram-types.md](references/diagram-types.md).

## 3. Workflow

1. **Read the code first, then draw.** Every box must be a real identifier
   (`ValidatedOrder`, `prepare_order`, `Groups::chunks`), every arrow a real
   call, ownership or transition. Cite `path:line` in the prose next to the
   diagram, not inside it.
2. **Draft small.** Aim for 8–25 nodes. Collapse fields into a summary line
   (`+FlexArray fees, fixed_fees, slippage…`) rather than listing all of them.
   If a class diagram needs more than ~12 classes, split by layer.
3. **Language.** Labels follow the document's language (Chinese in
   `*.zh-CN.md` and the alphax-docs knowledge base, English elsewhere); identifiers
   stay exactly as in the code. Mixed text is fine: `"① 价格：+inf→close"`.
4. **Style.** Use the house palette and stereotypes from
   [references/diagram-types.md](references/diagram-types.md) §Style
   (`<<enum>>`, `<<trait>>`, `classDef done/part/todo/hot`), and
   `direction LR|TB` explicitly on class diagrams.
5. **Validate before committing** (mandatory for any file with a
   ```mermaid block):

   ```bash
   python3 skills/mermaid-diagrams/scripts/check_mermaid.py docs/            # all docs
   python3 skills/mermaid-diagrams/scripts/check_mermaid.py path/to/file.md  # one file
   python3 skills/mermaid-diagrams/scripts/check_mermaid.py --png --out /tmp/mm file.md  # then look at the PNGs
   ```

   It prints `ok` / `FAIL file:line block #n (type)` with the parser's
   message and exits non-zero on any failure. Fix every failure; GitHub uses
   the same parser and shows an error box instead of the diagram.
6. **Look at the picture** when layout matters (`--png`, then open the file):
   a diagram that parses can still be unreadable (crossing arrows, 40 nodes in
   a row). Reorder declarations or split it.
7. **Cross-check against the code once more** after editing: a renamed
   function or a removed field makes a diagram wrong silently.

## 4. Syntax that breaks rendering

The full list with fixes is in
[references/syntax-pitfalls.md](references/syntax-pitfalls.md). The ones that
actually bit while writing this repo's docs:

- `::` (or any `:`) inside a **relation label** of a class diagram
  (`A --> B : Foo::bar`) — the colon is the label separator. Write `Foo.bar`.
- `{ }` inside a **class body** (`Pair{corr, window}`) — braces end the body.
  Write `Pair(corr, window)` or plain words.
- Parentheses in a class attribute line turn it into a method; that is
  cosmetic, but keep enum variants without parentheses if they should sit in
  the attribute compartment.
- `|`, `(`, `)`, `[`, `]`, `{`, `}` and `"` inside **flowchart node text** —
  wrap the text in quotes: `A["cash ≥ 0 | reject"]`; for a literal quote use
  `#quot;`.
- Generics: write `Vec~f64~`, never `Vec<f64>`.
- Line breaks: `<br/>` only inside quoted flowchart / sequence text; class
  members cannot break lines.
- `end` as a bare node id in a flowchart; `graph` vs `flowchart` (use
  `flowchart`); a `gantt` without `dateFormat`.

## 5. One-time setup for rendering

```bash
npm install @mermaid-js/mermaid-cli@11.4.2        # anywhere; the checker finds node_modules/.bin/mmdc upward
# Chromium: the checker looks at $PUPPETEER_EXECUTABLE_PATH, /opt/pw-browsers/*, ~/.cache/puppeteer,
# ~/.cache/ms-playwright and the usual system paths. Otherwise: npx puppeteer browsers install chrome
```

Without a local install the checker falls back to
`npx -y @mermaid-js/mermaid-cli@11.4.2` (needs network). Pin the version:
GitHub renders Mermaid 11; a newer local parser can accept syntax GitHub
rejects.

## 6. Where this fits with the rest of the docs

- [nivalume/alphax-docs](https://github.com/nivalume/alphax-docs) — the knowledge base, a separate repository;
  Mermaid everywhere. Its site renders the blocks with mermaid 11 in the browser;
  check a checkout with `python3 skills/mermaid-diagrams/scripts/check_mermaid.py ../alphax-docs`.
- The READMEs — Mermaid blocks only.
