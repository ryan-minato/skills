# ops

[English](README.md)

**平台操作**类 skill——每个协作平台一个整合的日常操作手册：issue、PR/MR、
CI、规划结构、发布、wiki、只读仓库调研，以及这些工作依赖的 CLI/MCP 工具链
配置。每个 skill 保持其平台的工具优先级（GitHub：MCP server 优先，其次
`gh`；GitLab：认证的 `glab` 优先，其次 GitLab Duo MCP server），在两者都
不可用时提供匿名只读 REST 兜底，并内置约定发现与发布前强制审查。设计目标
是小型本地模型也能执行：每个操作只给一条推荐路径，并全程使用决策表。
项目模板、标签与政策的创作属于 `engineering` catalog 的 community 类
skill。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| _（暂无）_ | |
