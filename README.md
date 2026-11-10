# alphax-agent

alphax-agent 是一个由 LLM 驱动的 alpha 研究与因子挖掘平台。研究循环（提出假设 → 设计实验 → 写代码 → 回测 → 评估）和编码搜索都实现为 LangGraph 图，LLM 调用统一走 litellm。设计参考 RD-Agent，核心抽象可替换：场景（scenario）、编码器（CoSTEER）、执行环境（environment）、工作流（workflow）、Agent。

当前内置场景：

- `qlib_factor`：在 qlib 数据上挖掘量价因子，因子写成 Python，用 LightGBM + TopkDropout 策略回测。
- `crypto_strategy`：为加密永续合约提出均值回归 / 趋势跟踪策略的经济学假设，并把假设设计成可回测的规则。目前只实现假设和实验设计两步，环境、编码、回测、总结和 baseline 留空。

## 架构

```
alphax_agent/
  config/      AlphaXSettings：YAML 运行文件 + .env + ALPHAX_AGENT_* 环境变量
  core/        领域模型与插件接口：Task、Workspace、Experiment、Trace、Feedback、
               五个研究角色（HypothesisProposer / ExperimentDesigner / Developer / Runner / Summarizer）、
               Scenario、ScenarioRegistry、PromotionRule、PromptLibrary
  llm/         LLMClient（重试、缓存、用量统计、审计日志）+ LiteLLMBackend + 结构化输出解析
  agents/      Agent 基类（prompt + tools + LLM → 经过校验的 pydantic 输出）、Tool、Toolbox
  env/         Environment 接口、LocalEnvironment（子进程，超时杀进程组）、QlibEnvironment
  coder/       CoSTEER：树搜索 + 反思经验库，LangGraph 实现
  workflow/    研究循环的 LangGraph 图、节点类、路由、ArtifactStore
  scenarios/
    qlib_factor/  数据导出、因子工作区、评估阶段、5 个 Agent、qlib 回测 Runner、prompts
    crypto_strategy/  数据 schema 渲染、假设和实验设计 2 个 Agent、prompts；其余角色是占位
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

凡是“prompt + 工具 + 调用 LLM”的组件都是 `Agent` 子类。子类声明 `name`、`output_model`（pydantic）、两个模板键，再提供一个有类型的方法（`propose`、`design`、`implement`、`review`、`summarize`）。基类负责：

- 渲染 Jinja2 模板，把输出 JSON schema 附在 system prompt 后；
- 工具调用循环（有轮数上限），工具参数用 pydantic 校验，工具报错以文本返回给模型；
- 解析失败或领域校验失败（`_validate` 或每次调用传入的 validator）时，把错误发回模型要求重答，有次数上限。

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

从 RD-Agent 的 crypto 场景移植而来，只保留“策略”这一支的两个提示词阶段。

| 环节 | 实现 |
|---|---|
| 数据描述 | `render_data_schemas` 读 `<data_root_dir>/schemas/*.json`，只描述 K 线（`market.crypto.bar`）和资金费率（`market.crypto_funding_rate`），且行数据必须在盘上；没有 schema 时提示词里写明“没有可用的数据描述” |
| 假设 | `StrategyHypothesisAgent`：每轮提出一个经济机制（均值回归或趋势跟踪），要求写清摩擦、对手方、为什么不被套利掉、承担的风险、可证伪的预测和失效状态；提示词不允许出现具体标的、交易所、日期，参数一律无量纲、时间以 bar 计 |
| 实验设计 | `StrategyDesignAgent`：把机制写成至多 `max_strategies_per_experiment` 个策略（一个机制及其变体）。每个策略只有名称和一段描述，描述里该写什么由提示词规定（LaTeX 的信号、仓位、闸门，持有期，可检验预测，参数含义等），代码里没有固定策略的结构。实验就是核心的 `Experiment` |
| 环境、编码、回测、总结、baseline | 未实现。`PendingDeveloper`、`PendingRunner`、`PendingSummarizer` 被调用时抛出 `ConfigurationError`，没有 baseline |

因为后面的角色是空的，用 `--until design` 只跑到实验设计：

```bash
alphax-agent run -c configs/crypto_strategy.yaml --until design
```

不加 `--until` 时，一轮会在 `develop` 步失败并记为失败轮，连续失败达到 `max_consecutive_failures` 后停止。

场景选项（`scenario.options`）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ALPHAX_AGENT_SCENARIO__OPTIONS__DATA_ROOT_DIR` | `~/jarvis-data` | 数据根目录：schema 在 `schemas/`，行数据在 `crypto/` 下；支持 `~` 和 `$VAR` |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__MAX_STRATEGIES_PER_EXPERIMENT` | `2` | 每个实验的策略数上限 |

## 安装

需要 Python 3.11+。qlib 需单独安装（例如从源码 `pip install -e /path/to/qlib`），并准备好 qlib 数据（如 `~/.qlib/qlib_data/cn_data`）。

```bash
pip install -e ".[dev]"
```

## 配置

配置有四个来源，优先级从高到低：

1. shell 环境变量
2. 当前目录的 `.env`
3. `-c` 指定的 YAML 运行文件（如 `configs/qlib_factor.yaml`）
4. 内置默认值

YAML 描述一次运行的完整参数（回测区间、搜索预算等）。`.env` 放密钥和与本机有关的值（模型、API key、qlib 路径）。`.env` 已在 `.gitignore` 中，不会被提交。

从模板创建 `.env`：

```bash
cp .env.example .env
```

至少设置模型和 API key：

```dotenv
ALPHAX_AGENT_LLM__MODEL=deepseek/deepseek-chat
ALPHAX_AGENT_LLM__API_KEY=sk-...
```

`.env.example` 列出常用的配置项并附说明，可选项默认注释掉。其余字段按下面的命名规则同样可以用环境变量设置。

### 变量命名

变量名 = `ALPHAX_` + 配置段 + `__` + 字段，层级之间用双下划线，不区分大小写。

| YAML 配置项 | 环境变量 |
|---|---|
| `llm.model` | `ALPHAX_AGENT_LLM__MODEL` |
| `workflow.max_loops` | `ALPHAX_AGENT_WORKFLOW__MAX_LOOPS` |
| `costeer.policy` | `ALPHAX_AGENT_COSTEER__POLICY` |
| `scenario.options.market` | `ALPHAX_AGENT_SCENARIO__OPTIONS__MARKET` |
| `scenario.options.qlib.provider_uri` | `ALPHAX_AGENT_SCENARIO__OPTIONS__QLIB__PROVIDER_URI` |

数字和布尔值直接写，字典和列表写成 JSON，例如 `ALPHAX_AGENT_LLM__EXTRA_PARAMS={"top_p": 0.9}`。覆盖场景选项时只改指定的那一项，YAML 里的其余选项保持不变。

### LLM（必填）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ALPHAX_AGENT_LLM__MODEL` | `gpt-4o-mini` | litellm 模型 id，一般是 `<服务商>/<模型名>` |
| `ALPHAX_AGENT_LLM__API_KEY` | 无 | API key |
| `ALPHAX_AGENT_LLM__API_BASE` | 无 | 网关或自建 OpenAI 兼容服务的地址 |
| `ALPHAX_AGENT_LLM__TEMPERATURE` | `0.2` | 采样温度 |
| `ALPHAX_AGENT_LLM__MAX_TOKENS` | 不限制 | 单次回复的 token 上限 |
| `ALPHAX_AGENT_LLM__TIMEOUT` | `180` | 单次请求超时，秒 |
| `ALPHAX_AGENT_LLM__MAX_RETRIES` | `4` | 临时错误（限流、超时、5xx）的重试次数 |
| `ALPHAX_AGENT_LLM__RETRY_BACKOFF` | `2.0` | 指数退避的底数，秒 |
| `ALPHAX_AGENT_LLM__JSON_MODE` | `true` | 模型支持时要求输出 JSON |
| `ALPHAX_AGENT_LLM__CACHE_DIR` | 关闭 | 响应缓存目录 |
| `ALPHAX_AGENT_LLM__EXTRA_PARAMS` | `{}` | JSON 对象，原样传给 `litellm.completion`（如 Azure 的 `api_version`） |

几种常见写法：

```dotenv
# DeepSeek
ALPHAX_AGENT_LLM__MODEL=deepseek/deepseek-chat
ALPHAX_AGENT_LLM__API_KEY=sk-...

# OpenAI
ALPHAX_AGENT_LLM__MODEL=openai/gpt-4o
ALPHAX_AGENT_LLM__API_KEY=sk-...

# OpenAI 兼容网关
ALPHAX_AGENT_LLM__MODEL=openai/<网关上的模型名>
ALPHAX_AGENT_LLM__API_BASE=https://my-gateway.example.com/v1
ALPHAX_AGENT_LLM__API_KEY=...
```

### qlib_factor 场景

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ALPHAX_AGENT_SCENARIO__OPTIONS__QLIB__PROVIDER_URI` | `~/.qlib/qlib_data/cn_data` | qlib 数据目录 |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__QLIB__PYTHON_BIN` | 运行 alphax-agent 的解释器 | 装有 qlib 的 Python，写绝对路径 |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__QLIB__TIMEOUT` | `3600` | 一次 qlib 回测的时间上限，秒 |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__DATA_DIR` | `~/.alphax/agent/data/qlib_factor` | 导出的行情数据目录 |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__MARKET` | `csi300` | 股票池 |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__BENCHMARK` | `SH000300` | 基准 |
| `ALPHAX_AGENT_SCENARIO__OPTIONS__USE_CRITIC` | `true` | 是否启用 LLM 代码审查评估阶段 |

研究循环和编码搜索的参数（`ALPHAX_AGENT_WORKFLOW__*`、`ALPHAX_AGENT_COSTEER__*`）用法相同，见 `.env.example`。完整字段见 `alphax_agent/config/settings.py` 和 `alphax_agent/scenarios/qlib_factor/settings.py`。

### 注意

- `.env` 从运行命令时的当前目录读取，请在仓库根目录执行 `alphax-agent`。
- 不要把 API key 写成空值（`ALPHAX_AGENT_LLM__API_KEY=`）。空字符串会被当作 key 发给服务商，并覆盖 shell 里导出的 `OPENAI_API_KEY` 等变量。不设置这一项时保持注释或删除这一行。
- 服务商自己的变量（`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`AZURE_API_BASE`、`AWS_REGION_NAME` 等）请在 shell 里 `export`。alphax-agent 只读取 `.env` 里 `ALPHAX_AGENT_` 开头的变量，写在 `.env` 里的其他变量不保证传到 litellm：litellm 自己的 `.env` 查找只在它安装在 `.env` 所在目录之下时才生效（例如项目内的 `.venv`）。只用一个 key 时，用 `ALPHAX_AGENT_LLM__API_KEY` 最简单。
- API key 不会写进运行目录，`settings.json` 中显示为 `**********`。

## 使用

1. 配置模型和 API key：`cp .env.example .env` 后填写（见上一节）。需要调整回测区间、股票池、搜索预算时，复制并修改运行文件 `configs/qlib_factor.yaml`。

2. 检查环境并导出数据（只需一次）：

```bash
alphax-agent prepare -c configs/qlib_factor.yaml
```

3. 运行研究循环：

```bash
alphax-agent run -c configs/qlib_factor.yaml --loops 10
```

4. 续跑、查看结果：

```bash
alphax-agent run -c configs/qlib_factor.yaml --resume alphax_runs/20261108-120000 --loops 5
```

```bash
alphax-agent report alphax_runs/20261108-120000
```

调试时可以只跑到某一步就停（`baseline`、`propose`、`design`、`develop`、`run`）：

```bash
alphax-agent run -c configs/qlib_factor.yaml --until propose
```

只跑一轮，该步结束后立即退出，之后的步骤不会启动；这一轮不写入 `loops/`，产出看日志。不能与 `--loops` 同用。

代码中使用：

```python
from alphax_agent.app import AlphaXApplication
from alphax_agent.config import AlphaXSettings

report = AlphaXApplication(AlphaXSettings.load("configs/qlib_factor.yaml")).run(loops=5)
print(report.render())
```

## 运行目录

```
alphax_runs/<run>/
  settings.json      本次运行的配置（密钥已脱敏）
  run.json           场景、开始/结束时间、进程号、状态（running / finished / failed / interrupted）、运行选项
  events.jsonl       每一轮每一步的开始、结束或失败，带耗时
  baseline.json      仅基础特征的回测结果
  loops/NNN.json     每轮的假设、实验（任务、代码、评估）、结果、反馈
  llm_calls.jsonl    每次 LLM 调用的 agent、请求、响应、推理内容、所属上下文、耗时
  summary.json       每轮指标与决策、各 agent 的 token 用量与成本
  workspaces/
    coding/<实验>/<任务>/<轮次_算子_节点>/   每次尝试的代码与输出；search_tree.json；lessons.json
    backtests/<实验>/                        qlib workflow.yaml、合并后的因子、mlruns
```

`llm_calls.jsonl` 每行除了请求和响应，还有两项供事后查看：

- `reasoning`：模型的推理内容。只有服务商返回时才有（litellm 的 `reasoning_content`，如 DeepSeek 的推理模型、Anthropic 的 thinking）；`response.usage.reasoning_tokens` 是其中花在推理上的 token 数。
- `context`：这次调用属于哪里。工作流内的调用有 `loop` 和 `stage`；agent 调用有 `conversation`（同一次 agent 调用的工具轮次、修复重答共用一个 id）和 `turn`；编码尝试里的调用还有 `task_id`、`task`、`node_id`（对应 `search_tree.json` 里的节点）、`operator`、`round`、`parent_id`。

`run.json` 在运行开始时写入、结束时更新。进程被强杀时它停在 `running`，此时可以用 `pid` 判断进程是否还在。`events.jsonl` 每行 `{time, loop, stage, event, seconds, ...}`，`event` 是 `started`、`finished` 或 `failed`（失败时带 `error_type` 和 `message`）。这两项和 `reasoning`、`context` 只在这些记录加入之后产生的 run 里有，旧 run 里没有。

配置 `llm.cache_dir` 后，相同请求从缓存返回，可在不调用 API 的情况下重放一次运行。缓存里的旧条目没有推理内容。

## 扩展

**新场景**：继承 `Scenario`，提供 `settings_model`、`context`（给 Agent 的场景描述）、`prepare()`、`assemble()`（返回五个角色）。在自己的包里声明 entry point 即可被发现，不需要改 alphax-agent：

```toml
[project.entry-points."alphax_agent.scenarios"]
my_scenario = "my_package.scenario:MyScenario"
```

**新执行后端**：实现 `Environment`（`prepare`、`run`、`python_command`），传给 `QlibEnvironment(settings, backend=...)`。

**新搜索策略**：继承 `SearchPolicy`，设置 `key` 并实现 `_choose`，配置 `costeer.policy: <key>`。

**新评估阶段**：继承 `EvaluationStage`，实现 `evaluate(context) -> StageResult`，加入场景的 `EvaluationPipeline`。

## 测试

```bash
pytest
```

pytest 从仓库根目录运行时也会读取那里的 `.env`。`.env` 里设置了 `ALPHAX_AGENT_WORKFLOW__MAX_LOOPS`、`ALPHAX_AGENT_COSTEER__POLICY` 这类运行参数时，`tests/test_workflow.py` 里的配置测试可能失败。跑测试前把这些行注释掉，只留模型和 API key。

`tests/test_qlib_e2e.py` 在合成的 qlib 二进制数据上跑真实的 qlib 流程（导出、编码搜索、前视检查、回测、晋升、去重），只有 LLM 是脚本化的；未安装 qlib 时跳过。
