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
| [scaffold-data-science](scaffold-data-science/) | 搭建可复现的 Python 数据科学项目：原始输入不可变、转换流水线带校验、数据产物记录自身来源，支持本地、S3 与 Hugging Face 存储。 |
| [scaffold-disposal](scaffold-disposal/) | 在取得新的明确确认后，先 dry-run 再移除复制安装的一次性 scaffold 技能，不触碰长期技能，也不越界删除其他 catalog 的构建技能。 |
| [scaffold-ml](scaffold-ml/) | 按「短期实验」或「长期训练代码库」两种形态搭建机器学习项目，包含硬件感知依赖、可复现运行、可选容器，并实时发现 GPU 基础镜像。 |
