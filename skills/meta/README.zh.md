# Meta Skills

用于创建持久 agent harness 的一次性、项目级构建技能。为 harness 构建安装此
catalog，将所有长期规则写入目标项目并完成验证，然后使用 `meta-disposal`
移除这些构建技能。统一的 description 前缀将其标识为临时 skill，因此该 catalog
可以为项目初始化提供完整指导，而不会在正常开发阶段持续占用上下文。

| Skill | 用途 |
|---|---|
| [meta-harness-building](meta-harness-building/) | 所有 harness 搭建、改进与修复的统一入口：先了解仓库，与用户敲定需求，计划获批后借助手头的 skill 逐层构建，在干净上下文中回读产物，审查只有构建技能才懂的词汇，让构建技能不进入任何提交，并在送审前询问是否移除它们。 |
| [meta-harness-architecture](meta-harness-architecture/) | 入口按层加载的架构实践手册：设计轴、entrypoint、knowledge、project skill、同步、熵治理、多 agent 拓扑与高级自治，并附起始形状资产。 |
| [meta-workflow-design](meta-workflow-design/) | 与人类开发者共同设计项目的平台无关管理模型——项目真正配得上的工作跟踪语义、贴合实际交付方式的 workflow profile、overlay 与变更传播风险——并落地 workflow 契约，供平台构建技能映射而非重新决定。 |
| [meta-agent-authority](meta-agent-authority/) | 设计人类-Agent 治理政策——H0–H3 权限级别、review admission 与 integration 两道关口、升级上报条件，以及 agent 永不自行扩权的规则——并将其作为项目 agent 运行时遵循的持久政策落地。 |
| [meta-github-workflow](meta-github-workflow/) | 为 GitHub.com 或 GitHub Enterprise 构建或系统修复完整的 GitHub 仓库生命周期 harness，围绕 PR 回路设计：intake 表单与 Discussions 路由、在默认集上扩展的标签、tracking issue 与 milestone、经关联分支的早期 draft PR、Actions 质量门与社区自动化、ruleset、CODEOWNERS、标签驱动发布说明的 release、注册表、可选 Projects 与 ML 实验记录，以及持久项目 agent 工作流。 |
| [meta-gitlab-workflow](meta-gitlab-workflow/) | 为 gitlab.com 或自托管实例构建或系统修复完整的 GitLab 项目生命周期 harness：规划、work item 与早期 draft MR、社区文件、CI/CD、治理、安全、Wiki、发布、部署、注册表、可选 MLOps 及持久项目 agent 工作流。 |
| [meta-git-branching](meta-git-branching/) | 依据项目的发布与部署实际选择 git 分支模型——git flow、GitHub Flow 或某个 GitLab Flow 变体——与团队敲定分支、tag、保护与合并方式约定，在平台上强制可强制的部分，并将该契约作为按事件触发的项目知识落地。 |
| [meta-gpu-container](meta-gpu-container/) | 为任意领域的项目建立 GPU 容器环境：判断是否需要容器，实时查证并选定当前的 CUDA/ROCm 基础镜像，为 docker run、Compose 与 dev container 接入 GPU，并将最终规则沉淀到目标 harness。 |
| [meta-disposal](meta-disposal/) | 在用户决定删除后，先 dry-run 再移除 `meta` 与 `scaffold` 两个 catalog 中复制安装的全部一次性构建技能，不触碰长期技能。 |
| [meta-python-defaults](meta-python-defaults/) | 在不替换现有工作选择的前提下，补全 Python 文档、测试与工具链约定。 |
