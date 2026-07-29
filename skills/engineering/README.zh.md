# engineering

[English](README.md)

软件工程类 skill：跨语言、跨框架适用的编程**方法论**（方法、工作流和实践），
以及不足以自立 catalog 的窄域**工件创作**工作流。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [code-refactoring](code-refactoring/) | 以测试保障的小步、保持行为不变的方式重构既有代码：把结构调整与行为变更分离，判断何时重构（何时不重构），诊断代码坏味道，并安全地执行标准的具名重构手法。 |
| [devcontainer-authoring](devcontainer-authoring/) | 创作、测试与发布 Dev Container 工件——Feature（install.sh 契约、幂等性与多基础镜像质量标准、独立性规则）、Template（选项替换、载荷设计、冒烟测试循环）与预构建镜像（devcontainer build --push、metadata 合并语义）——附带仓库脚手架与共享 action CI。 |
| [gitmoji](gitmoji/) | 起草 gitmoji 提交信息：先确定项目变体（独立语法 vs 叠加 CC 语法、unicode vs 文本代码），再通过首个匹配即停的决策列表为主要意图选出唯一 emoji，最后按交付前清单校验。 |
