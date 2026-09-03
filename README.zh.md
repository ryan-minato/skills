# Skills

[English](README.md)

一个精心维护的 [Agent Skills](https://agentskills.io) 库——自包含的指令包，
用于教会编码 agent（Claude Code、Codex、Copilot 及其他兼容客户端）
高质量地完成特定任务。

## 目录（Catalogs）

| Catalog | 内容 | 安装范围 |
|---|---|---|
| [`core`](skills/core/) | 推荐在所有环境安装的 skill | 全局（用户级） |
| [`engineering`](skills/engineering/) | 通用编程方法论类 skill 与窄域工件创作工作流（如 Dev Container 工件） | 按需安装到项目 |
| [`meta`](skills/meta/) | 通用 harness 机制的一次性构建技能：完整 agent harness、GitHub 与 GitLab 生命周期工作流、GPU 容器环境、Python 约定默认值 | 安装到项目，harness 验证后移除 |
| [`scaffold`](skills/scaffold/) | 特定主题项目（ML、数据科学、Colab 笔记本）的一次性构建技能——只安装匹配的那一个 | 安装到项目，harness 验证后移除 |
| [`writing`](skills/writing/) | 面向人类读者的写作：体裁类（学术、博客/评论、文案）与载体类（LaTeX、Typst、Markdown） | 按需安装到项目 |

每个 catalog 的 README 列出了其包含的 skill。

## 安装

### 单个 skill（skills CLI）

使用 [skills CLI](https://github.com/vercel-labs/skills) 安装单个 skill：

```bash
# 交互式选择 skill（项目级安装）
npx skills add ryan-minato/skills

# 安装指定 skill
npx skills add ryan-minato/skills --skill <skill-name>

# 全局安装（core 类 skill 推荐）
npx skills add ryan-minato/skills --skill <skill-name> -g
```

建议将 `core` 类 skill 全局安装，使其在所有项目中可用。其他 catalog 的 skill
按需安装到需要的项目中；`meta` 与 `scaffold` 包含临时构建技能，其长期产物
验证后应移除。

### 作为 Claude Code 插件

每个 catalog 也会作为独立的 Claude Code 插件发布。先添加一次 marketplace，
再按整个 catalog 安装，只启用需要的插件：

```
/plugin marketplace add ryan-minato/skills
/plugin install core@ryan-minato-skills
/plugin install meta@ryan-minato-skills     # 或 core@、engineering@、scaffold@、writing@ 等
```

## 参与贡献

约定、质量标准和仓库机制均已为人类和 agent 编写成文档：
从 [AGENTS.md](AGENTS.md) 开始，然后阅读
[ARCHITECTURE.md](ARCHITECTURE.md)。克隆后运行一次 `just setup`，
提交前运行 `just check`。

## 许可证

[Apache-2.0](LICENSE)
