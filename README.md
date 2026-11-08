# alphax

alphax 是一个由 LLM 驱动的 alpha 研究与因子挖掘平台。研究循环（提出假设 → 设计实验 → 写代码 → 回测 → 评估）和编码搜索都实现为 LangGraph 图，LLM 调用统一走 litellm。设计参考 RD-Agent，核心抽象可替换：场景（scenario）、编码器（CoSTEER）、执行环境（environment）、工作流（workflow）、Agent。

当前内置场景 `qlib_factor`：在 qlib 数据上挖掘量价因子，因子写成 Python，用 LightGBM + TopkDropout 策略回测。

## 架构

```
alphax/
  config/      AlphaXSettings：YAML 运行文件 + ALPHAX_* 环境变量
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
  app.py       组装入口 AlphaXApplication
  cli.py       命令行 alphax
```

### 研究循环（`alphax/workflow`）

```
START → baseline → propose → design → develop → run → summarize → record ─┐
                      ▲     任一步失败直接跳到 summarize，记为失败的一轮       │
                      └──────────────────── 下一轮 ◀──────────────────────────┘
```

- 每个节点是一个类（`ProposeNode`、`DevelopNode`……），包装一个研究角色；路由是 `StepRoute`、`LoopRoute`。
- 可恢复的错误（`AlphaXError` 子类）不会终止运行：该步返回 `StepFailure`，本轮以失败记录进历史，proposer 下一轮能看到原因。连续失败达到 `max_consecutive_failures` 时停止。
- 状态里的 `history` 是唯一的历史来源，`Trace` 是它的只读视图。
- 每轮写入 `loops/NNN.json`，可用 `--resume` 从运行目录续跑；也可以传 LangGraph checkpointer（配合 `CheckpointSerde.create()`）。

### Agent（`alphax/agents`）

凡是“prompt + 工具 + 调用 LLM”的组件都是 `Agent` 子类。子类声明 `name`、`output_model`（pydantic）、两个模板键，再提供一个有类型的方法（`propose`、`design`、`implement`、`review`、`summarize`）。基类负责：

- 渲染 Jinja2 模板，把输出 JSON schema 附在 system prompt 后；
- 工具调用循环（有轮数上限），工具参数用 pydantic 校验，工具报错以文本返回给模型；
- 解析失败或领域校验失败（`_validate` 或每次调用传入的 validator）时，把错误发回模型要求重答，有次数上限。

`qlib_factor` 里的 Agent：`FactorHypothesisAgent`、`FactorDesignAgent`、`FactorCoderAgent`、`FactorCriticAgent`、`FactorFeedbackAgent`，以及通用的 `ReflectionAgent`。

## CoSTEER：不用 RAG 的编码搜索（`alphax/coder`）

RD-Agent 的 CoSTEER 用向量检索找相似任务的成功/失败代码。alphax 换成三种不需要检索的机制：

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

## 执行环境（`alphax/env`）

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

## 安装

需要 Python 3.11+。qlib 需单独安装（例如从源码 `pip install -e /path/to/qlib`），并准备好 qlib 数据（如 `~/.qlib/qlib_data/cn_data`）。

```bash
pip install -e ".[dev]"
```

## 使用

1. 复制并修改运行文件 `configs/qlib_factor.yaml`，设置模型和 API key：

```bash
export ALPHAX_LLM__MODEL=deepseek/deepseek-chat
```

```bash
export ALPHAX_LLM__API_KEY=...
```

2. 检查环境并导出数据（只需一次）：

```bash
alphax prepare -c configs/qlib_factor.yaml
```

3. 运行研究循环：

```bash
alphax run -c configs/qlib_factor.yaml --loops 10
```

4. 续跑、查看结果：

```bash
alphax run -c configs/qlib_factor.yaml --resume alphax_runs/20261108-120000 --loops 5
```

```bash
alphax report alphax_runs/20261108-120000
```

代码中使用：

```python
from alphax.app import AlphaXApplication
from alphax.config import AlphaXSettings

report = AlphaXApplication(AlphaXSettings.load("configs/qlib_factor.yaml")).run(loops=5)
print(report.render())
```

## 运行目录

```
alphax_runs/<run>/
  settings.json      本次运行的配置（密钥已脱敏）
  baseline.json      仅基础特征的回测结果
  loops/NNN.json     每轮的假设、实验（任务、代码、评估）、结果、反馈
  llm_calls.jsonl    每次 LLM 调用的 agent、请求、响应、耗时
  summary.json       每轮指标与决策、各 agent 的 token 用量与成本
  workspaces/
    coding/<实验>/<任务>/<轮次_算子_节点>/   每次尝试的代码与输出；search_tree.json；lessons.json
    backtests/<实验>/                        qlib workflow.yaml、合并后的因子、mlruns
```

配置 `llm.cache_dir` 后，相同请求从缓存返回，可在不调用 API 的情况下重放一次运行。

## 扩展

**新场景**：继承 `Scenario`，提供 `settings_model`、`context`（给 Agent 的场景描述）、`prepare()`、`assemble()`（返回五个角色）。在自己的包里声明 entry point 即可被发现，不需要改 alphax：

```toml
[project.entry-points."alphax.scenarios"]
my_scenario = "my_package.scenario:MyScenario"
```

**新执行后端**：实现 `Environment`（`prepare`、`run`、`python_command`），传给 `QlibEnvironment(settings, backend=...)`。

**新搜索策略**：继承 `SearchPolicy`，设置 `key` 并实现 `_choose`，配置 `costeer.policy: <key>`。

**新评估阶段**：继承 `EvaluationStage`，实现 `evaluate(context) -> StageResult`，加入场景的 `EvaluationPipeline`。

## 测试

```bash
pytest
```

`tests/test_qlib_e2e.py` 在合成的 qlib 二进制数据上跑真实的 qlib 流程（导出、编码搜索、前视检查、回测、晋升、去重），只有 LLM 是脚本化的；未安装 qlib 时跳过。
