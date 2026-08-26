# ops

[English](README.md)

**GitHub 平台操作**——一个整合的日常操作手册，覆盖 issue、PR、CI、规划
结构、发布、Discussions、只读仓库调研、guardrail，以及这些工作依赖的
CLI/MCP 工具链配置。它优先使用 GitHub MCP server，其次使用 `gh`，并提供
匿名只读 REST 层和发布前强制审查。GitHub 模板、标签与政策创作属于
`engineering`；完整 GitLab 生命周期 harness 的构建属于一次性的 `meta`
catalog。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [github-ops](github-ops/) | 整合的 GitHub 操作：issue、PR、CI check 与 Actions 日志、Discussions、milestone/label/Projects、发布，以及对任意仓库的只读调研——MCP 优先、gh 兜底、匿名 REST 只读层、创建前模板发现、发布前强制审查；含 gh/MCP 配置与五个内置脚本 |
