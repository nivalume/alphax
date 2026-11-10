# 设计任务：alphax-agent 研究运行浏览器（TUI）

请为一个命令行工具设计一个**终端界面（TUI）**，用来浏览、复盘 LLM 驱动的 alpha 研究运行（下称 run）。
你没有访问代码库的权限，下面给出了你需要的全部背景、数据结构和真实样例。

## 1. 目标

命令：`alphax-agent tui [RUN_DIR]`

- 给了 `RUN_DIR`：直接打开这个 run。
- 没给：先进入 **run 选择器**，列出 `alphax_runs/` 下所有 run，选一个进入。任何时候都能回到选择器。
- **只读**。不发起、不续跑、不修改任何文件。
- **实时跟随**：一个 loop 要跑几分钟，run 运行期间文件在持续增长。界面要自动刷新，让用户看出现在卡在哪个阶段、
  哪次模型调用在等待、已经等了多久。已结束的 run 同样可浏览。
- 实现栈是 Python + Textual（可用 Rich 渲染 Markdown、语法高亮代码、Table、Tree、Tabs、DataTable、RichLog）。
  请只设计这些组件能做到的东西：不要图片、悬浮提示、自定义字体；LaTeX 无法真正渲染（见 5.5）。

用户是量化研究员，一边让 agent 跑、一边回看。他们要回答：
1. 这个 run 做了什么？哪些 loop 被晋升、哪些失败，失败在哪一步？
2. 某个 loop 里模型提出了什么假设、依据是什么？
3. 设计出的因子/策略的**公式**是什么？**代码**是什么？评估怎么判的？
4. 模型原样说了什么：完整 prompt、完整回复、**推理内容**、工具调用、修复/重试轮次、耗时、token、费用。
5. 回测结果和基线、和当前最佳比怎样。

## 2. 什么是 run

一个 run 由若干 **loop** 组成，每个 loop 依次经过这些**阶段**：

| 阶段 | 做什么 | 主要产物 |
|---|---|---|
| `propose` | 模型提出下一个假设 | hypothesis：statement、rationale、observation、justification、knowledge |
| `design` | 把假设设计成任务（因子或策略） | tasks：每个有 name、description；因子还有 formulation（LaTeX）和 variables |
| `develop` | 为每个任务写代码并评估。对每个任务做树搜索：节点 = 一份完整实现，operator 是 `draft`/`repair`/`improve`；每个节点依次过评估阶段 `execution` → `format` → `look_ahead` → `critic`，每个 PASS/FAIL/SKIPPED，带 0–1 分数和说明 | 代码文件、评估反馈、lessons |
| `run` | 回测 | metrics（IC、Rank IC、年化超额收益、最大回撤…）、notes、log |
| `summarize` | 固定规则决定是否晋升（promoted），模型写分析 | decision、reason、observations、hypothesis_evaluation、hypothesis_supported、next_hypothesis |

开头还有一个只执行一次的 `baseline` 步（可能为空）。**任一步失败会直接跳到 summarize**，这个 loop 记为失败：
`failure = {stage, error_type, message}`。连续失败几轮后 run 终止。

两个场景：
- `qlib_factor`：完整链路，有公式、代码、回测指标、搜索树。
- `crypto_strategy`：目前只有 propose 和 design；develop 之后是占位，**每个 loop 都以 “failed at develop” 结束，
  没有代码也没有指标**。任务的 description 是一大段自由文本，里面夹着 `$$…$$` 的 LaTeX。

界面必须同时适配这两个场景；缺失的内容要有**明确的空状态**（写出原因，如 “this scenario has no developer yet”），
不能留白。

## 3. 磁盘上的数据（设计依据）

标注：**[已有]** 所有 run 都有；**[新 run]** 后端补记之后产生的 run 才有，请按它们存在来设计，旧 run 里缺这些字段时显示 “not recorded”。

```
alphax_runs/<run_name>/                 run_name 默认是启动时间，如 20261109-214541
  settings.json        [已有] 配置快照（llm.model、scenario.name、workflow.max_loops 等，密钥已脱敏）
  run.json             [新 run] {scenario, started_at, finished_at|null, pid, status, options, resumes}
                       status: running | finished | failed | interrupted（running 但 pid 已不在 = 被强杀）
                       options: {loops, until, resume}；until 非空表示调试运行（--until）
  events.jsonl         [新 run] 步骤事件，每行 {time, loop, stage, event: started|finished|failed,
                       seconds（finished/failed 才有）, error_type?, message?}
                       stage 取 baseline | propose | design | develop | run | summarize | record；loop 从 0 开始
  baseline.json        [已有，可缺] ExperimentResult
  loops/NNN.json       [已有] 每个已结束的 loop 一份（运行中的 loop 还没有文件）
  llm_calls.jsonl      [已有] 每次模型调用一行，运行中追加，可达数 MB
  summary.json         [已有] 只在 run 正常结束时写；运行中或被中断的 run 没有它
  workspaces/…         [已有] 每次代码尝试的目录；search_tree.json；lessons.json；回测输出
```

`loops/NNN.json`（真实结构，字符串只列长度）：

```json
{
  "loop": 1,
  "hypothesis": {"statement": "(862字)", "rationale": "(1458字)", "observation": "(439字)",
                 "justification": "(260字)", "knowledge": "(298字)"},
  "experiment": {
    "id": "a1b2c3d4e5f6",
    "tasks": [{"id": "…", "name": "ECF_H2", "description": "(3362字，含 $$LaTeX$$)"}],
    "implementations": {
      "<task_id>": {
        "workspace": {"path": "…", "files": {"factor.py": "(源码)"}},
        "feedback": {"accepted": true, "score": 0.98,
                     "stages": [{"stage": "execution", "passed": true, "score": 1.0, "message": "…", "skipped": false}]},
        "artifacts": {"included": "true"}
      }
    },
    "result": {"metrics": {"IC": 0.031, "Rank IC": 0.044,
                           "1day.excess_return_with_cost.annualized_return": 0.08},
               "notes": ["…"], "log": "…"}
  },
  "feedback": {"decision": false, "reason": "…", "observations": "…", "hypothesis_evaluation": "…",
               "hypothesis_supported": null, "next_hypothesis": "…",
               "failure": {"stage": "develop", "error_type": "ConfigurationError", "message": "…"}}
}
```

失败的 loop：`implementations` 为空、`result` 为 null、`failure` 非空。

`llm_calls.jsonl` 一行：

```json
{"time": "2026-11-09T13:46:20+00:00", "agent": "strategy_proposer", "seconds": 36.3,
 "request": {"messages": [{"role": "system", "content": "(11k字)"}, {"role": "user", "content": "(1.2k字)"}],
             "tools": null, "json_mode": true},
 "response": {"content": "(JSON 字符串)", "tool_calls": [], "model": "openai/deepseek-v4.1-flash",
              "finish_reason": "stop", "cached": false,
              "usage": {"prompt_tokens": 2431, "completion_tokens": 2589, "cost": 0.0, "reasoning_tokens": 0}},
 "error": null,
 "reasoning": "(推理内容，可能数千字，也可能为空)",
 "context": {"loop": 1, "stage": "propose", "conversation": "7f3a9c1e", "turn": 0}}
```

`reasoning`、`usage.reasoning_tokens` 和 `context` 是 **[新 run]**，其余 [已有]。`reasoning` 只有服务商返回推理内容时才有，
否则为 null（不是错误，显示 “no reasoning returned”，与旧 run 的 “not recorded” 区分）。`context` 的字段：
- 所有工作流内的调用：`loop`、`stage`。
- agent 调用：`conversation`（一次 agent 调用一个 id）、`turn`（从 0 递增）。同一个 `conversation` 里的多条记录是工具轮次或
  “输出不合法，要求重答”的修复轮次，后一条的 messages 是前一条的延伸——请把它们设计成**一组**展示。
- develop 阶段里 coder / critic 的调用还有：`task_id`、`task`（任务名）、`node_id`（搜索树节点）、`operator`
  （draft/repair/improve）、`round`、`parent_id`（父节点，draft 为 null）。reflector 的调用只有 `round`。
- 不在工作流内产生的调用，`context` 为空对象。

`cached: true` 表示命中缓存（耗时 0 秒，没有真实调用；缓存里的旧条目没有推理内容）。

agent → 阶段：`*_proposer`→propose；`*_designer`→design；`factor_coder`、`factor_critic`、`reflector`→develop；
`factor_analyst`→summarize。

`summary.json` 里有按 agent 汇总的 `usage: {calls, prompt_tokens, completion_tokens, cost}`。

### 真实样例（用来填充你的设计稿，请直接使用）

假设（crypto_strategy，loop 0）：

> statement: Large multi-bar displacements in perpetual futures are caused by forced deleveraging: liquidation engines and clustered stop orders execute market orders, pushing price beyond the level justified by voluntary flow, so part of the move is given back once forced orders clear. The state is the N-bar cumulative return normalized by the instrument's recent realized volatility; when |z| exceeds 2, enter contrarian … for h bars.
>
> rationale: The transfer persists because the contrarian position is painful to hold exactly when the signal fires … Falsifiable prediction: conditional on |z_N|>2 … decaying to insignificance after roughly h=4-8 bars …

任务 `FFRev_H4` 的 description（节选）：

```
Phenomenon: mean_reversion.
Mechanism: Forced deleveraging overshoot cleared by liquidity providers.
Signal: $$ z_t = \frac{\sum_{i=0}^{N-1} r_{t-i}}{\sigma_t \sqrt{N}}, \quad r_t = \ln(C_t/C_{t-1}) $$
Position: $$ P_t = -\operatorname{sgn}(z_t)\,\mathbf{1}\{|z_t|>z_c\}, \quad P_t\in[-1,1]. $$
Horizon: Enter at the close of the bar where the signal fires, hold for h bars, exit at the close of bar t+h.
Gate: none.
Testable prediction: Conditional on |z_t|>z_c, the forward h-bar log return has sign -sign(z_t) …
```

失败 loop 的 feedback.reason：`The develop step failed with ConfigurationError: The crypto_strategy scenario has no developer yet; stop the run after the design stage with --until design`。

qlib_factor 的任务名如 `REV5`，formulation 如 `-\frac{1}{5}\sum_{i=0}^{4} r_{t-i}`，指标键如 `IC`、`Rank IC`、
`1day.excess_return_with_cost.annualized_return`。

规模参考：一个 run 约 5–50 个 loop；每个 loop 1–3 个任务；每个任务的搜索最多约 6 次尝试 × 5 轮；
单条 prompt 消息可达 1 万字以上；propose 一次调用约 30–40 秒，design 约 80 秒。

## 4. 导航结构

三层，每层都能一键回到上一层，任何地方 `?` 打开快捷键帮助、`q` 退出：

1. **Run 选择器** → 2. **Run 总览**（loop 列表） → 3. **Loop 详情**（阶段时间线 + 阶段内容）。

另有两个**全局视图**，在 run 内任何位置都能进、能回到原位置：**LLM 调用日志**、**产物视图**（公式 / 代码 / 推理）。

## 5. 需要设计的界面

### 5.1 Run 选择器
- 一行一个 run，最近的在上。列：名称/开始时间、场景、**状态**（running / finished / failed / interrupted / debug(--until)）、
  loop 数、晋升数、最后失败的阶段、模型、总 token、费用、时长。运行中的 run 用动态标记并显示已运行时长。
- 过滤和排序（按场景、状态；按时间/费用）。`/` 搜索。
- 高亮行时右侧或下方有**预览**：配置要点（模型、max_loops、场景选项）、每个 loop 一个小方块组成的状态条
  （晋升/未晋升/失败于某阶段/进行中）、最近一个假设的第一句。
- 空状态：`alphax_runs/` 不存在或为空；某个 run 目录损坏（缺 settings.json、jsonl 有半行）要能降级显示，不能崩。

### 5.2 Run 总览（loop 列表）
- 顶栏：run 名、场景、模型、状态、已用时、token/费用（可按 agent 展开）、基线指标（没有则显示 “no baseline”）。
- 主体：loop 表，列为 loop 号、**结果**（PROMOTED / not promoted / failed at <stage> / running at <stage>）、假设一句话、
  任务名、头部指标（IC、年化超额…，没有则 “—”）、耗时、token。
- 能看出趋势：指标随 loop 的走势（用字符 sparkline 或迷你条形即可）、晋升点。
- 选中 loop 回车进入详情；右侧预览该 loop 的假设与反馈要点。

### 5.3 Loop 详情
- 顶部是**阶段时间线**：`propose → design → develop → run → summarize`，每段显示状态（done / failed / skipped / running / pending）、
  耗时、该阶段的模型调用数与 token。失败的阶段高亮并显示错误类型；失败之后被跳过的阶段标 skipped。
- 下方是所选阶段的内容，**用 Tab 分页**，各阶段至少包含：
  - **Output**：解析后的产物，格式友好（不是原始 JSON）。propose 展示假设的五个字段；design 展示任务列表；
    develop 展示每个任务的搜索树和选中的实现；run 展示指标表；summarize 展示决策和分析。
  - **Reasoning**：模型的推理内容（长文本，纯文本；可折叠段落、可搜索）。
  - **Prompt**：system / user / 后续轮次的消息，逐条可折叠；长消息默认折叠并显示字数。
  - **Response**：原始回复（JSON 高亮）、tool_calls、finish_reason。
  - **Calls**：该阶段所有调用，按 conversation 分组（工具轮次、修复轮次嵌在组内），每行：agent、turn、耗时、token、费用、
    cached / error 标记。
  - 有错误时多一个 **Error** 页：错误类型、消息、发生在哪次调用。
- **develop 阶段的专有设计**（qlib_factor）：
  - 每个任务一棵**搜索树**（Tree 组件）：节点 = 尝试，标注 operator（draft/repair/improve）、轮次、分数、是否 accepted；
    父子关系清楚。
  - 选中节点：**代码**（多文件，语法高亮，行号）、与父节点的 **diff**、**评估阶段列表**（execution/format/look_ahead/critic，
    PASS/FAIL/SKIPPED + 分数 + 消息）、coder 的 rationale、对应的模型调用（含推理）。
  - 该实验的 **lessons**（跨任务经验）单独一页。
- **run 阶段**：指标表（metric | baseline | 当前最佳 | 本次 | 变化），变好/变差有颜色**和**符号；notes；log（可滚动）。
- **summarize 阶段**：晋升规则的判定结果（promoted / not promoted + 原因）与模型分析并排，分清 “规则判的” 和 “模型说的”。

### 5.4 全局视图：LLM 调用日志
按时间顺序列出整个 run 的所有调用（可达数百条）：时间、loop、阶段、agent、turn、耗时、token、费用、状态。
过滤：loop、阶段、agent、仅错误、仅 cached、仅慢调用（> N 秒）。回车打开一次调用的完整内容（Prompt / Reasoning / Response）。
运行中新调用出现在底部，**等待中的调用**显示已等待秒数（尚无 response）。

### 5.5 全局视图：产物（公式 / 代码 / 推理）
让用户不用逐层点进去，就能把 run 里**同一类产物**放在一起看和对比：
- **Formulas**：所有任务的公式。因为终端不能渲染 LaTeX，请设计**两种显示**并给出推荐：(a) 带语法高亮的 LaTeX 源码；
  (b) 转成 Unicode/ASCII 的近似排版（如 `z_t = Σ r_{t-i} / (σ_t√N)`），可一键切换。公式要和所属任务、loop 关联，
  能按任务名搜索，能并排对比两个任务的公式。
- **Code**：所有实现的文件树 + 代码查看，支持在同一任务的不同尝试之间做 diff。
- **Reasoning**：按阶段/调用浏览所有推理内容，支持全文搜索。

### 5.6 实时跟随
- 默认对运行中的 run 开启跟随：新的调用、阶段、loop 出现时自动刷新；用户手动滚动或选中内容时**不要抢焦点**，
  改为在角落提示 “N new — press f to follow”。`f` 切换跟随。
- 清楚区分 “进行中 / 等待模型 / 已结束 / 进程已退出但没写完（interrupted）”。
- 刷新不能让当前选中项、滚动位置、展开/折叠状态丢失。

### 5.7 缺失与异常状态（每一种都要出示意）
旧 run 没有 reasoning/context（显示 “not recorded”，调用按 agent 名和时间顺序归入阶段并标注 “inferred”）；没有 summary.json；
没有 baseline；crypto_strategy 的 develop 起全部为空；失败 loop 没有 implementations/result；jsonl 末尾半行；
超长消息（几万字）；窗口过窄。

## 6. 约束

- 键盘优先，鼠标可选。给出完整**键位表**：方向键 + vim 风格 `hjkl`、`Enter` 进入、`Esc`/`Backspace` 返回、
  `Tab`/`Shift+Tab` 切页、`[` `]` 上/下一个 loop、`1`–`5` 跳到阶段、`/` 搜索 + `n`/`N`、`f` 跟随、`y` 复制当前内容、
  `w` 换行开关、`o` 原始 JSON 视图、`?` 帮助、`q` 退出。可调整，但要成体系、不冲突，并在界面底部常驻当前上下文可用的键。
- 目标尺寸 **120×40**，必须在 **80×24** 下可用（说明窄屏时如何降级：隐藏预览、折叠列、改为单列）。
- 颜色：支持 truecolor，降级到 256 色；同时给 **深色和浅色**两套。**状态不能只靠颜色**（同时用符号/文字）：
  promoted ✓、failed ✗、running ◐/spinner、skipped –、cached ⟳。建议一套语义色令牌（success / failure / running / muted /
  accent / diff-add / diff-del / warning）。
- 性能：`llm_calls.jsonl` 可达数 MB，单条消息可达数万字——设计要允许**懒加载、折叠、分页**，并给出加载中的样子。
- 所有文本里可能有中文和英文混排（假设、说明可能是中文），宽字符对齐要考虑。
- 界面标签用简短英文；内容原样显示，不翻译。

## 7. 请交付

1. **界面清单**：每个屏幕/状态一张终端示意图（等宽字符画，包含边框、颜色标注说明），用上面的真实样例填充；
   至少包含：Run 选择器（含运行中、含空状态）、Run 总览、Loop 详情的每个阶段页（propose Output、design Output + 公式、
   develop 搜索树 + 代码 + 评估、run 指标、summarize）、Reasoning 页、Prompt 页、Calls 页、全局调用日志、
   Formulas 视图（两种显示）、Code 视图（含 diff）、实时跟随提示、各缺失/异常状态。每张图给出 120×40 版本，
   关键屏幕再给 80×24 版本。
2. **导航与状态图**：屏幕之间怎么跳转、返回到哪里、全局视图如何进出并回到原位置。
3. **键位表**：全局键 + 每个屏幕的专有键。
4. **组件对应**：每个区域建议用的 Textual / Rich 组件，以及需要自定义的部分。
5. **视觉规范**：语义色令牌（深/浅两套）、符号表、间距与对齐规则、焦点/选中/禁用的表现。
6. **交互细节**：实时刷新时的行为、折叠/展开规则、搜索、复制、窄屏降级。
7. 对每个你做了取舍的地方，用一两句话说明理由，并列出你认为需要和我确认的问题。

## 8. 不在范围内

发起/续跑/停止 run；编辑配置、代码或 prompt；图表库式的可视化（最多 sparkline）；鼠标专属交互；网络访问。
