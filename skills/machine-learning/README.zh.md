# machine-learning

[English](README.md)

面向**训练或评估模型的项目在日常工作中**长期使用的项目级 skill：记录一次
run 实际使用了什么、把一系列假设组织成一个 research task、塑造实验代码与
其配置面、决定训练 run 应输出哪些信号以及何时告警、诊断异常的 run。按项目
需要安装其中的若干个；每个 skill 可独立工作，并按角色把相关工作交接给
同 catalog 的其他 skill。

初始化这类项目——目录布局、环境、命令、检查、agent 指南——属于一次性的
`scaffold` catalog（`scaffold-ml`）；GPU 容器环境属于一次性的 `meta`
catalog。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skill 列表

| Skill | 说明 |
|---|---|
