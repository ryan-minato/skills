# scaffold

[English](README.md)

面向**特定主题项目**的一次性、项目级构建技能——这类项目应当包含什么，输入、
代码与产物如何组织，其 agent 继承哪些约定。只安装**一个**：与你要构建的项目
相匹配的那一个；它们互为替代，而非可叠加的层。将所有长期规则写入目标项目并
完成验证，然后使用 `meta` catalog 的 `meta-disposal` 移除这些构建技能。

可与其中任意一个叠加使用的通用 harness 机制——完整 harness 架构、GitHub 与
GitLab 生命周期工作流、Python 约定默认值——位于一次性的 `meta` catalog。项目
初始化完成后，两个 catalog 由 `meta-disposal` 一起移除。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [scaffold-colab](scaffold-colab/) | 搭建以 Google Colab 笔记本为交付物的 Colab 中心项目：根目录笔记本与真实 Colab 互为镜像、官方 Colab runtime 镜像 devcontainer 做本地初验、colab-mcp 连接真实会话，并附一套可读笔记本写作准则。 |
| [scaffold-data-science](scaffold-data-science/) | 搭建可复现的 Python 数据科学项目：原始输入不可变、转换流水线带校验、数据产物记录自身来源，支持本地、S3 与 Hugging Face 存储。 |
| [scaffold-ml](scaffold-ml/) | 搭建单一形态的可复现机器学习项目：带每次 run 解析转储的类型化配置面、由用户选择的依赖载体（默认 uv project，或 uv 编译的 requirements）、带 run manifest 与 tracker 的显式 Accelerate 循环、绑定已记录基准的评估入口、可选的容器配方（镜像 digest 即环境身份）、实验级的测试/类型/hook 规则、研究任务约定，以及按角色指向长期机器学习技能的 agent 指南。 |
