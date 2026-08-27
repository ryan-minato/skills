# scaffold

[English](README.md)

面向**特定主题项目**的一次性、项目级构建技能——这类项目应当包含什么，输入、
代码与产物如何组织，其 agent 继承哪些约定。只安装**一个**：与你要构建的项目
相匹配的那一个；它们互为替代，而非可叠加的层。将所有长期规则写入目标项目并
完成验证，然后使用 `scaffold-disposal` 移除这些构建技能。

可与其中任意一个叠加使用的通用 harness 机制——完整 harness 架构、GitHub 与
GitLab 生命周期工作流、Python 约定默认值——位于一次性的 `meta` catalog。项目
初始化完成后，两个 catalog 通常一起移除。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
