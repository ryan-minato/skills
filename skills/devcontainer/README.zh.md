# devcontainer

[English](README.md)

Dev Container **创作**类 skill——为 [Dev Container 生态](https://containers.dev)
开发 Feature、创建 Template、预构建镜像。如需*使用* dev container
（为项目创建开发环境），请使用 `core` catalog 中的 `devcontainer-setup` skill。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [devcontainer-feature-authoring](devcontainer-feature-authoring/) | 开发、测试并发布 Dev Container Feature：manifest schema 与 install.sh 契约、质量标准（幂等性、基础镜像容忍度、非 root 正确性、确定性安装）、feature 独立性规则，以及带共享 action CI 的现代仓库脚手架。 |
| [devcontainer-image-prebuild](devcontainer-image-prebuild/) | 预构建并发布 dev container 镜像：devcontainer build --push、devcontainer.metadata 标签及其与消费端配置的合并语义、瘦消费端配置、devcontainers/ci 的 CI 自动化、cacheFrom 层复用与 tag 策略。 |
| [devcontainer-template-authoring](devcontainer-template-authoring/) | 创建、测试并发布 Dev Container Template：manifest schema 与 templateOption 替换语义、载荷与选项设计、默认值替换式冒烟测试循环、发布流程，以及带共享冒烟测试 action 的现代仓库脚手架。 |
