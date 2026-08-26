# engineering

[English](README.md)

通用编程**方法论**类 skill——跨语言、跨框架适用的方法、工作流和实践——
外加不足以自立 catalog 的窄域**工件创作**工作流（如 Dev Container 工件与
持久化视觉设计规范）。构建 GitHub 或 GitLab 项目的完整生命周期
harness——包括协作文件、规范与日常平台工作流——属于一次性的 `meta`
catalog。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [code-refactoring](code-refactoring/) | 以测试保障的小步、保持行为不变的方式重构既有代码：把结构调整与行为变更分离，判断何时重构（何时不重构），诊断代码坏味道，并安全地执行标准的具名重构手法。 |
| [devcontainer-authoring](devcontainer-authoring/) | 创作、测试与发布 Dev Container 工件——Feature（install.sh 契约、幂等性与多基础镜像质量标准、独立性规则）、Template（选项替换、载荷设计、冒烟测试循环）与预构建镜像（devcontainer build --push、metadata 合并语义）——附带仓库脚手架与共享 action CI。 |
| [design-md](design-md/) | 创作并校验持久化、供 agent 读取的 DESIGN.md 视觉设计规范，包含可选 YAML 设计 token、正文指导、上游格式检查和 OKLCH 计算器。 |
| [gitmoji](gitmoji/) | 起草 gitmoji 提交信息：先确定项目变体（独立语法 vs 叠加 CC 语法、unicode vs 文本代码），再通过首个匹配即停的决策列表为主要意图选出唯一 emoji，最后按交付前清单校验。 |
| [goal-alignment](goal-alignment/) | 与用户对齐创建物（软件、系统、实验、skill、服务等）应达成的目标：以一轮轮追问推进直至共识（可推断处附建议答案，仅用户可知的事实则直接提问），再把共识记录为单一事实来源的目标文档——整体目标、带验证方式与层级（硬约束/优化目标/偏好）的具体目标、分级要求（强制/尽力/偏好）与权衡决策记录。只谈目标；不含计划与架构。 |
