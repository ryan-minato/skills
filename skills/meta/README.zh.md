# Meta Skills

用于创建持久 agent harness 的一次性、项目级构建技能。为 harness 构建安装此
catalog，将所有长期规则写入目标项目并完成验证，然后使用 `meta-disposal`
移除这些构建技能。统一的 description 前缀将其标识为临时 skill，因此该 catalog
可以为项目初始化提供完整指导，而不会在正常开发阶段持续占用上下文。

| Skill | 用途 |
|---|---|
| [meta-harness-architecture](meta-harness-architecture/) | 调查、规划、构建、审计并维护完整 harness，包括渐进加载、反馈循环、同步与熵治理。 |
| [meta-gitlab-workflow](meta-gitlab-workflow/) | 为 gitlab.com 或自托管实例构建或系统修复完整的 GitLab 项目生命周期 harness：规划、work item 与早期 draft MR、社区文件、CI/CD、治理、安全、Wiki、发布、部署、注册表、可选 MLOps 及持久项目 agent 工作流。 |
| [meta-disposal](meta-disposal/) | 在取得新的明确确认后，先 dry-run 再移除复制安装的一次性构建技能，不触碰长期技能。 |
| [python-project-defaults](python-project-defaults/) | 在不替换现有工作选择的前提下，补全 Python 文档、测试与工具链约定。 |
| [ml-project-scaffold](ml-project-scaffold/) | 搭建短期 ML 实验或长期训练项目，并实时发现 GPU 镜像。 |
| [data-science-project-scaffold](data-science-project-scaffold/) | 搭建输入不可变、产物具备 provenance 的可复现 Python 数据科学项目。 |
