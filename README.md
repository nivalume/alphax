# alphax-agent

alphax-agent 是一个由 LLM 驱动的 alpha 研究与因子挖掘平台。研究循环（提出假设 → 设计实验 → 写代码 → 回测 → 评估）和编码搜索都实现为 LangGraph 图，LLM 调用统一走 litellm。设计参考 RD-Agent，核心抽象可替换：场景（scenario）、编码器（CoSTEER）、执行环境（environment）、工作流（workflow）、Agent。

当前内置场景：

- `qlib_factor`：在 qlib 数据上挖掘量价因子，因子写成 Python，用 LightGBM + TopkDropout 策略回测。
- `crypto_strategy`：为加密永续合约提出均值回归 / 趋势跟踪策略的经济学假设，设计成规则，写成目标仓位代码，在 jfinance 数据上用 alphax 回测。

## 架构

```
alphax_agent/
  config/      AlphaXSettings：YAML 运行文件 + .env + ALPHAX_AGENT_* 环境变量
  core/        领域模型与插件接口：Task、Workspace、Experiment、Trace、Feedback、
               五个研究角色（HypothesisProposer / ExperimentDesigner / Developer / Runner / Summarizer）、
               Scenario、ScenarioRegistry、PromotionRule、PromptLibrary
  llm/         LLMClient（重试、缓存、用量统计、审计日志）+ LiteLLMBackend + 结构化输出解析
  agents/      Agent 基类（prompt + tools + LLM → 经过校验的 pydantic 输出）、AgentPrompt、Tool、Toolbox
  env/         Environment 接口、LocalEnvironment（子进程，超时杀进程组）、QlibEnvironment、AlphaXEnvironment
  coder/       CoSTEER：树搜索 + 反思经验库，LangGraph 实现
  workflow/    研究循环的 LangGraph 图、节点类、路由、ArtifactStore
  scenarios/
    qlib_factor/  数据导出、因子工作区、评估阶段、5 个 Agent 及各自的 prompt 类、qlib 回测 Runner
    crypto_strategy/  jfinance schema 渲染与面板导出、策略工作区与 harness、评估阶段、6 个 Agent 及各自的 prompt 类、alphax 回测 Runner
  app.py       组装入口 AlphaXApplication
  cli.py       命令行 alphax-agent（兼容 alphax 别名）
```

### 研究循环（`alphax_agent/workflow`）

```
START → baseline → propose → design → develop → run → summarize → record ─┐
                      ▲     任一步失败直接跳到 summarize，记为失败的一轮       │
                      └──────────────────── 下一轮 ◀──────────────────────────┘
```

- 每个节点是一个类（`ProposeNode`、`DevelopNode`……），包装一个研究角色；路由是 `StepRoute`、`LoopRoute`。
- 可恢复的错误（`AlphaXError` 子类）不会终止运行：该步返回 `StepFailure`，本轮以失败记录进历史，proposer 下一轮能看到原因。连续失败达到 `max_consecutive_failures` 时停止。
- 状态里的 `history` 是唯一的历史来源，`Trace` 是它的只读视图。
- 每轮写入 `loops/NNN.json`，可用 `--resume` 从运行目录续跑；也可以传 LangGraph checkpointer（配合 `CheckpointSerde.create()`）。

### Agent（`alphax_agent/agents`）

凡是“prompt + 工具 + 调用 LLM”的组件都是 `Agent` 子类。子类声明 `name`、`output_model`（pydantic），接收自己的 prompt 对象，再提供一个有类型的方法（`propose`、`design`、`implement`、`review`、`summarize`）。基类负责：

- 用 prompt 渲染 system 和 user 消息，把输出 JSON schema 附在 system prompt 后；
- 工具调用循环（有轮数上限），工具参数用 pydantic 校验，工具报错以文本返回给模型；
- 解析失败或领域校验失败（`_validate` 或每次调用传入的 validator）时，把错误发回模型要求重答，有次数上限。

每个 Agent 的 prompt 是它自己的一个类，继承 `AgentPrompt`（即 `langchain_core` 的 `BaseChatPromptTemplate`），告诉模型什么、怎么写都由这个类决定。一个 prompt 由三层组成：

- **YAML 段**：场景 `prompts.yaml` 里的 `system` / `user` 模板（如 `proposer.system`），以及几个 Agent 共用的段落（如 `scenario.background`）。模板自己决定各段放在哪里、用什么标题。
- **prompt 类里的函数**：每个段落一个私有方法（如 `_background()`、`_experiment_setting()`），从类的字段（场景设置、数据描述等）渲染出来；`sections()` 返回这个 prompt 用到的段落，`inputs` 声明每次调用时由 Agent 传入的变量（历史、假设等）。
- **通用渲染**：`AgentPrompt.format_messages(**inputs)` 合并段落、LangChain 的 partial 变量和调用参数，经 `PromptLibrary.render` 渲染（保留 `trim_blocks` 和 `StrictUndefined`）。`invoke({...})` 会检查 `inputs` 是否齐全。

同一场景的几个 prompt 可以共用一个基类放共有段落（如 `FactorPrompt`、`CryptoPrompt`），但每个 prompt 只取自己需要的段落。通用的 `CoderAgent` / `ReflectionAgent` 的 `CoderPrompt` / `ReflectorPrompt` 不描述任何领域；场景用自己的 prompt 类加上段落，并在自己的 `prompts.yaml` 里覆盖 `coder.system` / `reflector.system`。

`qlib_factor` 里的 Agent：`FactorHypothesisAgent`、`FactorDesignAgent`、`FactorCoderAgent`、`FactorCriticAgent`、`FactorFeedbackAgent`，以及通用的 `ReflectionAgent`。

## CoSTEER：不用 RAG 的编码搜索（`alphax_agent/coder`）

RD-Agent 的 CoSTEER 用向量检索找相似任务的成功/失败代码。alphax-agent 换成三种不需要检索的机制：

1. **每个任务一棵解树**（`SolutionTree`）。节点是一份完整实现及其评估结果，边是产生它的算子：
   - `draft`：从零写；
   - `repair`：修复被拒的父节点，提示词里有父节点代码、完整评估反馈和祖先尝试摘要；
   - `improve`：改进已接受但分数未达标的节点。
2. **搜索策略**（`SearchPolicy`，可注册新策略）决定每轮扩展哪个节点：
   - `greedy`：总是扩展最好的节点；
   - `ucb`：`score + c·sqrt(ln N / (1 + n))`，被反复扩展的节点让位给少探索的兄弟节点；
   - 被拒节点只在作为分支叶子时修复，且连续失败少于 `max_repair_depth` 次；没有可扩展节点时重新 draft；达到 `target_score` 或预算用完即停止。
3. **经验库**（`LessonBook`）。每轮结束后，`ReflectionAgent` 读取本轮有信息量的转移（失败被修好、失败持续、首稿失败、已接受版本退化），写出与具体任务无关的规则。每条经验带确定性的错误签名（失败的评估阶段 + 异常类型，如 `execution:KeyError`）。给编码器选经验时按“签名匹配度 → 被确认次数 → 新近程度”排序，不做向量检索。经验库跨实验保留，配置 `lesson_store` 后跨运行保留。

此外，同一实验中已被接受的兄弟任务实现会作为“已验证的约定”给到编码器（协作）。

一轮的 LangGraph 图：

```
schedule ──Send(每个未解决任务)──▶ attempt（并行：写代码 → 评估）──▶ integrate（入树 → 反思）──▶ schedule …
```

评估是分阶段的 `EvaluationPipeline`：前一阶段失败后面的阶段跳过；确定性阶段在前，LLM 阶段在后，大多数拒绝不消耗 token。分数是各阶段加权和，全部通过才算接受。每个实验的搜索树写入 `search_tree.json`。

## 执行环境（`alphax_agent/env`）

- `Environment`：在工作目录里执行命令，返回 `EnvResult`。
- `LocalEnvironment`：本地子进程，每个命令独立进程组，超时连同子进程一起杀掉。
- `QlibEnvironment`：在某个后端（默认 `LocalEnvironment`，可换成容器或远程沙箱实现）之上提供 qlib 能力：检查 qlib 可导入和数据目录、读取交易日历、运行 `qrun`、从 recorder 导出指标。
- `AlphaXEnvironment`：同样组合一个后端，检查 alphax 和 pyarrow 可导入、jfinance 数据目录和 schema 目录存在。设置是 `AlphaXEnvSettings`（解释器、数据目录、市场、频率、schema 列表）。

每个场景的选项都继承框架的 `ScenarioSettings`，它只有一个字段 `environment`（`qlib` 或 `alphax`）；场景用 `supported_environments` 声明自己能在哪些环境里运行，选了不支持的环境会在加载配置时报 `ConfigurationError`。回测区间、数据、费用这些因场景而异的选项由各场景自己定义：`qlib_factor` 用 `data_start` / `data_end` / `segments`，`crypto_strategy` 用 `period`。

## qlib_factor 场景

| 环节 | 实现 |
|---|---|
| 数据 | `FactorDataExporter` 在 qlib 环境里导出 `full`（全市场全区间，回测用）、`debug`（少量股票短区间，编码用）、`debug_cut`（`debug` 在某日截断，前视检查用）三份 `daily_pv.h5` |
| 因子接口 | `factor.py` 读 `daily_pv.h5`，写 `result.h5`：索引 (`datetime`, `instrument`)，一列，列名为因子名 |
| 评估阶段 | `execution`（能跑、有输出）→ `format`（索引、列名、dtype、无 inf、NaN 比例、截面有区分度、与输入对齐）→ `look_ahead`（在截断数据上重跑，截断日之前的值必须不变，抓负向 shift、居中窗口、全样本标准化等）→ `critic`（LLM 代码审查，可关闭） |
| 回测 | 已接受因子在全样本上计算 → 与因子库按平均截面相关系数去重（默认阈值 0.99）→ 基础特征 + 因子库 + 新因子进入 LightGBM → TopkDropout 回测 |
| 晋升 | `PromotionRule`：主指标（默认 IC）必须提升，守护指标（超额年化收益、最大回撤）恶化不超过容差。决策由规则做，LLM 只负责分析和建议下一步，结果可复现、可审计 |

因子库 = 被晋升实验中实际进入回测的因子。

## crypto_strategy 场景

| 环节 | 实现 |
|---|---|
| 环境 | 只支持 `alphax`。`AlphaXEnvironment` 用 `alphax.python_bin` 指定的解释器（在 alphax 仓库 `uv sync` 后的 `.venv/bin/python`） |
| 数据描述 | `render_data_schemas` 读 `<data_dir>/<market>/schemas/*.json`，只描述 `alphax.schemas` 里列出的、是 K 线数据集（`stream_fields` 以 `bar_type` 为键，有 `ts_open`/`ts_close` 和 OHLCV）且有该频率行数据的数据集；没有时提示词里写明“没有可用的数据描述” |
| 数据导出 | `StrategyDataExporter` 在 alphax 环境里用 `alphax.data` 读 jfinance 的 parquet（`<slug>.v<N>/<INSTRUMENT>-1-DAY-LAST-EXTERNAL/*.parquet`，decimal 转 float64，bar 在收盘后 1 ms 可用），写三份 `(bars, instruments)` 面板：`full`（预热 + 全区间，回测用）、`debug`（少量标的、样本内最后若干根，编码用）、`debug_cut`（`debug` 截断，前视检查用）。标的池默认按样本内成交额中位数取前 N 个，只用 `oos_start` 之前的数据 |
| 假设 | `StrategyHypothesisAgent`：每轮提出一个经济机制（均值回归或趋势跟踪），要求写清摩擦、对手方、为什么不被套利掉、承担的风险、可证伪的预测和失效状态；提示词不允许出现具体标的、交易所、日期，参数一律无量纲、时间以 bar 计 |
| 实验设计 | `StrategyDesignAgent`：把机制写成至多 `max_strategies_per_experiment` 个策略（一个机制及其变体），每个策略是名称加一段描述 |
| 策略接口 | `strategy.py` 定义 `PARAMS` 和 `positions(data, **params)`：`data` 是字段名到 `(bars, instruments)` 数组的 dict，返回同形状、取值 [-1, 1] 的目标仓位（NaN 为空仓）。可用 numpy 和 `alphax.factors` 表达式。平台自带的 `run_strategy.py` 负责加载、延迟、费用和回测，不在 Agent 可改的文件里 |
| 评估阶段 | `execution`（能跑、产出仓位）→ `format`（形状、无 inf、在 [-1, 1] 内、有持仓、会变化）→ `look_ahead`（在截断面板上重跑，截断前的仓位必须不变）→ `critic`（LLM 代码审查，可关闭） |
| 回测 | 每个通过的策略在 `full` 上用 `alphax.from_orders` 回测：一个现金共享账户，标的 i 的目标权重为 position_i / N（N 为当根有 bar 的标的数），只在目标变化时下单，按 `signal_delay` 延迟成交。指标分 `is.*`（样本内）和样本外两套；实验结果取主指标最好的那个策略 |
| 样本外 | 只有 `is.*` 进入结果、历史和 LLM 提示词并参与晋升；样本外指标写在回测目录的 `oos_metrics.json`，结果的 `artifacts["oos_metrics"]` 指向它 |
| 晋升 | `PromotionRule`：默认主指标 `is.portfolio.sharpe`，守护指标 `is.portfolio.max_drawdown`（容差 0.05）。没有 baseline，第一个结果直接成为当前最优 |

```bash
alphax-agent prepare -c configs/crypto_strategy.yaml   # 检查环境并导出面板
alphax-agent run -c configs/crypto_strategy.yaml --loops 1
```

场景选项（`scenario.options`，完整示例见 `configs/crypto_strategy.yaml`）：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `environment` | `alphax` | 只支持 `alphax` |
| `alphax.python_bin` | 当前解释器 | 装有 alphax 和 pyarrow 的解释器；相对路径以当前目录为准，不解析 venv 的符号链接 |
| `alphax.data_dir` / `alphax.market` | `~/.jfinance` / `crypto` | jfinance 数据目录；支持 `~` 和 `$VAR` |
| `alphax.interval` | `1d` | 目前只支持日频，其他值报错 |
| `alphax.schema_dir` | `<data_dir>/<market>/schemas` | schema 目录 |
| `alphax.schemas` | `[binance.vision.market.crypto.bar.raw]` | 提供给 Agent 的 K 线数据集；第一个是主数据集，其余按 bar 对齐 |
| `period.start` / `oos_start` / `end` | `2021-01-01` / `2025-01-01` / `2026-11-01` | 样本内 `[start, oos_start)`，样本外 `[oos_start, end)` |
| `universe.symbols` / `max_instruments` / `min_bars` | 自动 / `30` / `365` | 显式标的，或按样本内流动性选 |
| `backtest.fees` / `slippage` / `init_cash` / `signal_delay` / `warmup_bars` | `0.0005` / `0.0002` / `10000` / `1` / `180` | 成本、资金、延迟、预热 |
| `debug.instruments` / `bars` / `cut_fraction` | `5` / `500` / `0.75` | 编码用的小面板 |
| `cache_dir` | `~/.alphax/agent/data/crypto_strategy` | 导出面板的位置；导出配置变化时自动重新导出 |
| `max_strategies_per_experiment` | `2` | 每个实验的策略数上限 |

`run.json` 在运行开始时写入、结束时更新。进程被强杀时它停在 `running`，此时可以用 `pid` 判断进程是否还在。`events.jsonl` 每行 `{time, loop, stage, event, seconds, ...}`，`event` 是 `started`、`finished` 或 `failed`（失败时带 `error_type` 和 `message`）。这两项和 `reasoning`、`context` 只在这些记录加入之后产生的 run 里有，旧 run 里没有。

配置 `llm.cache_dir` 后，相同请求从缓存返回，可在不调用 API 的情况下重放一次运行。缓存里的旧条目没有推理内容。

## 扩展

**新场景**：继承 `Scenario`，提供 `settings_model`、`prepare()`、`assemble()`（返回五个角色，并为每个 Agent 构造它自己的 prompt）。在自己的包里声明 entry point 即可被发现，不需要改 alphax-agent：

```toml
[project.entry-points."alphax_agent.scenarios"]
my_scenario = "my_package.scenario:MyScenario"
```

**新执行后端**：实现 `Environment`（`prepare`、`run`、`python_command`），传给 `QlibEnvironment(settings, backend=...)` 或 `AlphaXEnvironment(settings, backend=...)`。

**新运行环境**：在 `env/` 下实现一个组合后端的 `Environment` 及其设置类，把名字加进 `core/scenario.py` 的 `EnvironmentName`，并在支持它的场景的 `supported_environments` 里列出。

**新搜索策略**：继承 `SearchPolicy`，设置 `key` 并实现 `_choose`，配置 `costeer.policy: <key>`。

**新评估阶段**：继承 `EvaluationStage`，实现 `evaluate(context) -> StageResult`，加入场景的 `EvaluationPipeline`。

## 测试

```bash
pytest
```

pytest 从仓库根目录运行时也会读取那里的 `.env`。`.env` 里设置了 `ALPHAX_AGENT_WORKFLOW__MAX_LOOPS`、`ALPHAX_AGENT_COSTEER__POLICY` 这类运行参数时，`tests/test_workflow.py` 里的配置测试可能失败。跑测试前把这些行注释掉，只留模型和 API key。

`tests/test_qlib_e2e.py` 在合成的 qlib 二进制数据上跑真实的 qlib 流程（导出、编码搜索、前视检查、回测、晋升、去重），只有 LLM 是脚本化的；未安装 qlib 时跳过。

`tests/test_alphax_e2e.py` 在合成的 jfinance 数据上跑真实的 alphax 流程（导出、编码搜索、前视检查、回测、样本内外拆分、晋升），只有 LLM 是脚本化的。解释器取 `$ALPHAX_PYTHON`，否则取同级 alphax 仓库的 `.venv/bin/python`；都没有 alphax 时跳过。用 `-m alphax` 或 `-m "not alphax"` 选择。
