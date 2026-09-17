# sdd

[English](README.md)

规格驱动开发按三层拆分，各层独立演进：**方法论**
（`spec-driven-development`——实践本身、specify–clarify–approve–tasks–implement–verify
循环、审批包、规格如何与 issue 及 pull/merge request 配合），以及**每个框架一个**
skill（`openspec-workflow`、`spec-kit-workflow`——该工具的记录、其审批包、request
ready 前的完成判据、以及按平台安装的 request 自动化）。项目自身的规则——级别、
工具、审批模式、存档执行者——由一次性的 `meta` catalog 的 `meta-spec-workflow`
与用户商定并沉淀，它保持框架无关，把工具采用与自动化交给这里的框架 skill。

安装方法论 skill 和项目所用框架的 skill；每个都可单独工作，并按角色交接给同伴。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [spec-driven-development](spec-driven-development/) | 以工具与平台无关的方式从书面规格出发工作：判断规格驱动开发何时值得、采用哪个级别，在项目契约之下运行循环——规格写好即发布草稿，补齐审批包（规格加上有必要时的受约束 design）后停下，门主宣布讨论收口后核对，此后才写任务，验证，在 request ready 前于其内部存档——没有契约时按文档化的默认值执行，验收标准只保留在规格中，把原型或棕地代码库逐个变更地转换过来而不回填规格，并把项目所用框架交给对应的 `sdd` skill、把项目规则交给 `meta` catalog 的构建技能。 |
| [openspec-workflow](openspec-workflow/) | 通过 pull/merge request 运行 OpenSpec 变更：审批包（proposal、delta spec、有必要时的 design；tasks 在审批后）、spec-less 标记、严格校验器的运行时机、在 request 内部手工或由 `spec/archive` 标签机器人存档、`/spec show` 与 `/spec status` 评论命令、存档轴与进度轴状态标签，以及把它们装进 GitHub（ready 的 request 含未存档变更即失败的检查，评论、标签、状态三个 workflow）或 GitLab（标签门控与手动作业）的自动化；附带 `spec_changes.py`。 |
| [spec-kit-workflow](spec-kit-workflow/) | 通过 pull/merge request 运行 Spec-Kit 特性：审批包（规格与计划；tasks 在审批后）、ready 前的完成判据是任务全部勾选（该套件没有存档操作）、覆盖触及特性的 `/spec` 评论命令与进度标签，以及把它们装进 GitHub 或 GitLab 的自动化；附带 `spec_kit_features.py`。 |
