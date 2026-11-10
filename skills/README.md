# skills/

仓库级的 Agent Skills（[agentskills.io 规范](https://agentskills.io/specification)）。
每个子目录一个技能，入口是 `SKILL.md`，可选 `scripts/`、`references/`、`assets/`。
这里是**唯一的正本**；各编码助手通过下面的目录发现它们。

| 技能 | 用途 |
|---|---|
| [`docs-canvas/`](docs-canvas/SKILL.md) | 把架构说明、API 参考、walkthrough 等文档做成可导航的 Cursor Canvas（依赖 Cursor Canvas SDK） |
| [`mermaid-diagrams/`](mermaid-diagrams/SKILL.md) | 在 Markdown 里画类图 / 时序图 / 流程图 / 状态图 / 路线图，并用 mermaid-cli 逐块渲染校验 |
| [`alphax-strategy/`](alphax-strategy/SKILL.md) | 用 alphax 的 Python API 写策略、扫参数、检验结果是否成立 |
| [`alphax-feature/`](alphax-feature/SKILL.md) | 给引擎加功能：代码放哪里、必须守住的约束、测试和检查的顺序、收尾 |
| [`alphax-testing/`](alphax-testing/SKILL.md) | 黄金 fixture 测试：重放 fixture、判断失败原因、加覆盖、没有 fixture 时的性质和变异测试 |

## 各工具怎么发现这些技能

| 工具 | 项目级发现路径 | 本仓库的做法 | 用户级路径（跨仓库） |
|---|---|---|---|
| Claude Code | `.claude/skills/<name>/SKILL.md`（沿目录向上到仓库根；跟随符号链接） | `.claude/skills/mermaid-diagrams` → 符号链接到 `skills/mermaid-diagrams` | `~/.claude/skills/` |
| OpenAI Codex CLI / IDE | `.agents/skills/<name>/SKILL.md`（`$CWD` 向上到仓库根；跟随符号链接）；用 `$mermaid-diagrams` 或 `/skills` 显式调用 | `.agents/skills/mermaid-diagrams` → 符号链接 | `~/.agents/skills/`、`/etc/codex/skills` |
| xAI Grok Build | `.grok/skills/<name>/SKILL.md`（向上到仓库根）；也读取 Claude Code 的技能目录；`grok inspect` 可查看发现结果 | `.grok/skills/mermaid-diagrams` → 符号链接 | `~/.grok/skills/`、`~/.agents/skills/` |

三个工具都读根目录的 `AGENTS.md`（Claude Code 经 `CLAUDE.md` 里的 `@AGENTS.md` 导入），里面有一段指向这些技能和校验命令。

## 符号链接

`.claude/skills`、`.agents/skills`、`.grok/skills` 下的条目是指向 `skills/` 的相对符号链接，
所以技能只维护一份。Linux / macOS 开箱即用；Windows 需要 `git config core.symlinks true`
并以允许创建符号链接的身份检出，否则会看到一个内容为路径的文本文件。那种情况下运行：

```bash
./skills/install.sh --copy --project      # 用复制代替链接
```

## 安装到用户级目录（在其他仓库里也能用）

```bash
./skills/install.sh                 # 链接到 ~/.claude/skills ~/.agents/skills ~/.grok/skills
./skills/install.sh --copy          # 复制而不是链接
./skills/install.sh --uninstall     # 移除
```

## 新增技能

1. `skills/<name>/SKILL.md`，frontmatter 至少含 `name`（与目录同名，小写、数字、连字符）和 `description`（说清做什么、什么时候用）。
2. 在 `.claude/skills`、`.agents/skills`、`.grok/skills` 各加一条符号链接：`ln -s ../../skills/<name> .claude/skills/<name>` 等。
3. 在本文件的表格和根目录 `AGENTS.md` 里加一行。
4. 可选：`npx skills-ref validate skills/<name>` 校验 frontmatter。
