# Diagram types: when to use each, and templates that render

All templates below were rendered with mermaid-cli 11.4.2 (the parser GitHub
uses). Copy the skeleton, replace the identifiers, keep the structure.

## Style (shared)

Palette, shared by every diagram in the repository and the knowledge base:

| role | fill | stroke | use |
|---|---|---|---|
| plain | `#f6f8fa` | `#8c959f` | default boxes |
| accent (基础 / 强调) | `#ddf4ff` | `#0969da` | foundations, the thing the diagram is about |
| ok / done | `#dafbe1` | `#1a7f37` | finished, top layer |
| warn / hot (难点) | `#ffebe9` | `#cf222e` | the hard or risky step |
| purple / shell | `#fbefff` | `#8250df` | façades, external boundary |
| amber / partial | `#fff8c5` | `#9a6700` | partially done |

```mermaid
flowchart LR
  A["plain"] --> B["accent"] --> C["done"] --> D["hot"]
  classDef base fill:#ddf4ff,stroke:#0969da;
  classDef done fill:#dafbe1,stroke:#1a7f37;
  classDef hot fill:#ffebe9,stroke:#cf222e;
  classDef part fill:#fff8c5,stroke:#9a6700;
  classDef shell fill:#fbefff,stroke:#8250df;
  class B base; class C done; class D hot;
```

Stereotypes in class diagrams: `<<enum>>`, `<<trait>>`, `<<mod>>`, `<<fn>>`.
State `direction LR` or `TB` explicitly. Keep one idea per diagram.

## classDiagram — types and their relationships

Use for: ownership (`*--`), aggregation (`o--`), implementation (`<|..`),
inheritance (`<|--`), "uses / creates" (`..>`), plain association (`-->`).

```mermaid
classDiagram
  direction TB
  class Order {
    +f64 size = +inf
    +f64 price = +inf
    +SizeType size_type
    +bool allow_partial = true
  }
  class ValidatedOrder {
    -Order order
    -Option~&'static str~ violation
    +new(Order)$
    +set_bar(size, price, size_type, direction)
    +check() Result
  }
  class RecordSink { <<trait>> +push_order(row, col, size, price, fees, side) }
  class OrderRecords { +Vec~i64~ col, idx; +Vec~f64~ size, price, fees; +concat(parts)$ }
  class SizeType { <<enum>> Amount; Value; Percent; TargetAmount; TargetValue; TargetPercent }

  ValidatedOrder *-- Order
  Order --> SizeType
  RecordSink <|.. OrderRecords
  ValidatedOrder ..> OrderRecords : execute_validated → push_order
```

Rules of thumb:

- Members: `+` public, `-` private, `$` static after the member, `Vec~T~` for
  generics, a comment after a member goes in parentheses or Chinese
  parentheses（…）but never `{}`.
- Several members on one line separated by `;` inside the braces is fine and
  keeps big classes short.
- Relation labels after ` : ` may not contain another `:`; use `.` for paths.
- Enum variants without parentheses land in the attribute compartment (what
  you want); with parentheses they are drawn as methods (harmless but odd).

## sequenceDiagram — one unit of work, step by step

Use for: "what happens in one bar / one request / one call", especially when
several participants take turns.

```mermaid
sequenceDiagram
  participant G as simulate_group
  participant C as cash.rs
  participant K as execute_validated
  participant S as Sink
  Note over G: 组初始化：cash = init_cash[0, 首列]
  loop 每根 K 线 i
    G->>C: at_open：入金 → 分红
    C->>S: push_cash(Deposit / Dividend)
    Note over G: 第一遍：解析 price / val_price；group_value
    loop 按 seq 顺序的每列 k
      G->>K: set_bar(...)；ProcessOrderState{...}
      K-->>G: (exec_state, result)
      alt 成交
        G->>S: push_order(i, col, ...)
      end
      opt log
        G->>S: push_log(LogEntry{...})
      end
    end
    G->>C: earn：收盘收支
  end
```

Rules of thumb: `->>` call, `-->>` return, `Note over A,B:` for state, `loop` /
`alt` / `opt` / `par` blocks; keep ≤ 6 participants; put formulas in notes, not
in message text.

## flowchart — decisions and fixed stage order

Use for: a function with a fixed order of checks (`prepare_order`,
`execute_validated`), a pipeline, a dependency graph (`flowchart BT`).

```mermaid
flowchart TD
  A["① 价格：+inf→close；−inf→本根 open"] --> B{"use_stops 且 stops.armed()？"}
  B -- 是 --> C["先测 SL，再测 TP，测完再 trail"] --> D
  B -- 否 --> D{"止损触发了？"}
  D -- 是 --> E["from_stop → 信号由持仓方向决定"]
  D -- 否 --> F["read_signals → 解冲突链"]
  E --> G["signals_to_size"]
  F --> G
  G --> H{"size == 0？"}
  H -- 是 --> N["返回 None"]
  H -- 否 --> P["Pending{...}"]
  classDef hot fill:#ffebe9,stroke:#cf222e;
  class C hot;
```

Rules of thumb: always quote node text (`["…"]`, `{"…"}`) so `|()[]{}` and
`<br/>` are safe; edge labels with `-- 文本 -->` or `-->|文本|`; number the
stages (①②③) when the order is the point; `classDef` + `class` for the
palette; `subgraph` for layers.

Dependency graph variant (`flowchart BT`, arrows mean "depends on"):

```mermaid
flowchart BT
  core["alphax-core"] 
  exec["alphax-exec"] --> core
  sim["alphax-sim"] --> exec
  classDef base fill:#ddf4ff,stroke:#0969da;
  class core base;
```

## stateDiagram-v2 — lifecycle of one thing

Use for: a protocol, an order's life, a position's stop state.

```mermaid
stateDiagram-v2
  [*] --> pending : propose
  pending --> registered : approve（contract_check）
  registered --> run : run
  run --> rejected : gate 未过
  run --> pass : gate 通过
  pass --> oos : oos（≤ 3 批）
  oos --> forward : forward
  forward --> retired : z < −2
```

Rules of thumb: transition labels after ` : `; no colons inside; keep ≤ 12
states; use `state X { ... }` for nesting only when it clarifies.

## gantt — a plan in time

Use for: reading plans, roadmaps with durations. Tip: `dateFormat D` with
`axisFormat 第%d天` gives "day N" labels without real dates.

```mermaid
gantt
  title 阅读计划
  dateFormat  D
  axisFormat  第%d天
  section 准备
  环境与全局地图        :a0, 1, 1d
  section 交易路线
  alphax-core + alphax-model :a1, after a0, 1d
  alphax-sim               :a3, after a1, 3d
```

Rules of thumb: tasks need an id to be referenced by `after`; sections group
rows; keep under ~15 rows.
