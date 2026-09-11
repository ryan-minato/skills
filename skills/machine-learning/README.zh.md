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
| [experiment-provenance](experiment-provenance/) | 记录并判定一次 run 的身份——实际执行的源码快照、解析后的完整配置、环境身份（镜像或 lock 摘要加主机事实）、输入身份，以及与 commit 区分的 run id——保持 run 历史不可改写，并接入 tracker（已有 → 平台自带 → Trackio）来保存不含密钥的 manifest。 |
| [research-workflow](research-workflow/) | 端到端运行一个 research task：带 objective 与 evaluation 的研究 spec、一个 task 对应一个 PR/MR 并容纳多个假设、在隔离分支上带快照 commit 的假设循环、与声明相匹配的证据、在搜索空间与算力允许时的自动搜索，以及包含负结果在内的收尾结论。 |
| [experiment-code-conventions](experiment-code-conventions/) | 塑造实验代码：只抽象语义耦合、容忍偶然相似，优先成熟的第一方依赖并连同来源 vendor 不稳定的研究代码，保持显式训练循环，让配置面只承载一次 run 可选择的值，用轻量的 CPU 默认测试集测试行为契约、GPU 专属测试在无硬件时失败，hook 不跑测试，接近默认的 lint 且不对 tensor 代码设全局类型门，并在热路径保住性能、在其他地方恢复可理解性。 |
