# core

[English](README.md)

推荐**全局（用户级）安装**的 skill——无论在什么项目中都有用。

```bash
npx skills add ryan-minato/skills --skill <skill-name> -g
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [agentic-writing](agentic-writing/) | 编写、审查并精简 agent 阅读的文档——入口点文件、知识库、agent 面向的规范格式：把上下文花在改变行为上，按加载形态而非主题拆分，以触发条件措辞上下文指针，用 leading word 锚定行为，以正向表述取代禁令，每个含义只留一个权威来源，并按预期最弱的模型精简。 |
| [conventional-commits](conventional-commits/) | 起草符合 Conventional Commits 1.0.0 规范的 git 提交信息：规则优先级（文档 > commitlint 配置 > 历史 > 默认值）、首个匹配即停的 type 决策列表、scope 与破坏性变更策略，以及交付前校验清单。 |
| [devcontainer-setup](devcontainer-setup/) | 在可信来源策略下创建与修改 dev container 配置（mcr.microsoft.com/devcontainers、NVIDIA NGC、ghcr.io/devcontainers、ghcr.io/stacit-ai），内置来源枚举脚本、非预构建镜像的基线 feature 规则，以及 NVIDIA/AMD GPU 指引。 |
| [git-commit](git-commit/) | 以有序门禁执行完整的 git 提交工作流：按明确优先级发现项目约定、检查变更原子性、扫描暂存 diff 中的机密与 PII、核对提交者身份、运行 hooks 与本地检查，并在提交前用内置脚本校验提交信息。 |
| [great-skill-writing](great-skill-writing/) | 编写并改进行为可预测的 Agent Skill：触发准确的 description、渐进式披露、subagent 辅助的行为评估、隔离的脚本测试，以及内置校验脚本。 |
| [human-writing](human-writing/) | 以英文、中文或日文起草、修改和审读面向人类读者的文本：先明确类型/作者/读者/预期效果，按长度预算列大纲，保留作者立场，规避可识别的 AI 写作模式，核实引用，并克制地修改。 |
| [meta-harness](meta-harness/) | 将核心方法论应用于所有 harness 相关行动：先调研，以 AGENTS.md 作为渐进加载地图，按项目校准各层厚度，闭合前馈与反馈循环，并管理长期 harness 熵增。 |
| [plan-clarification](plan-clarification/) | 追问一个计划、想法或决定，直到 agent 与用户达成同一理解：将其建模为决策树，每轮把当前可问的问题连同推荐答案通过宿主自带的提问工具抛出，所有事实自己去查，且不在工作区留下任何文件。 |
| [programming-guidelines](programming-guidelines/) | 应用通用编程工作标准：编码前先思考，优先选择简单方案，保持改动精确，并用清晰的成功标准验证结果。 |
| [ryan-minato-skills-installing](ryan-minato-skills-installing/) | 将 ryan-minato/skills 库中的 skill 安装到项目或全局：优先通过现代包运行器（pnpm/bun/yarn）或 npx 使用 vercel-labs skills CLI，在没有 Node 环境时回退到内置的纯标准库克隆复制脚本，并可发现列出可用的 skill。 |
| [sensitivity-check](sensitivity-check/) | 检测文本或文件中的 PII 与泄露的机密信息，并生成结构化报告。首选引擎（经 uv 运行的 Presidio、detect-secrets）配合纯标准库回退脚本，覆盖通用及美/英/中/日实体。 |
