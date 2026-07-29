# engineering

[English](README.md)

通用编程**方法论**类 skill——跨语言、跨框架适用的方法、工作流和实践——
外加平台**社区文件创作**类 skill：编写定义仓库协作方式的文件,包括
issue/PR 模板、标签体系、提交与发布规范、CI 校验,以及社区健康文件
(CONTRIBUTING、CODE_OF_CONDUCT、SECURITY 等);另有不足以自立 catalog 的
窄域**工件创作**工作流(如 Dev Container 工件)。community 类 skill 负责
制定政策与结构;日常的平台操作属于 `ops` catalog。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [code-refactoring](code-refactoring/) | 以测试保障的小步、保持行为不变的方式重构既有代码：把结构调整与行为变更分离，判断何时重构（何时不重构），诊断代码坏味道，并安全地执行标准的具名重构手法。 |
| [devcontainer-authoring](devcontainer-authoring/) | 创作、测试与发布 Dev Container 工件——Feature（install.sh 契约、幂等性与多基础镜像质量标准、独立性规则）、Template（选项替换、载荷设计、冒烟测试循环）与预构建镜像（devcontainer build --push、metadata 合并语义）——附带仓库脚手架与共享 action CI。 |
| [gitmoji](gitmoji/) | 起草 gitmoji 提交信息：先确定项目变体（独立语法 vs 叠加 CC 语法、unicode vs 文本代码），再通过首个匹配即停的决策列表为主要意图选出唯一 emoji，最后按交付前清单校验。 |
| [github-community](github-community/) | 为 GitHub 仓库编写协作文件：issue 表单与同步的标签体系、PR 模板与 CONTRIBUTING 规则、commit 规范(附内置 stdlib 校验器与 CI workflow)、版本政策与 release.yml、社区健康文件(CODE_OF_CONDUCT、SECURITY、SUPPORT、GOVERNANCE、FUNDING.yml、组织级 .github 默认仓库),以及生成的项目级 skill。 |
| [gitlab-community](gitlab-community/) | 为 GitLab 项目编写协作文件(gitlab.com 与自管实例)：带快捷指令的 issue/MR 描述模板、附同步脚本的 scoped 标签体系、含 Changelog trailer 与 tokenless MR 流水线校验的 commit 规范、版本政策与 changelog_config.yml 及 tag 流水线检查、社区文件(CONTRIBUTING、CODE_OF_CONDUCT、SECURITY),以及生成的项目级 skill。 |
