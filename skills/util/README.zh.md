# util

[English](README.md)

作用于**工作过程本身**的 skill——agent 与用户如何一起澄清、追问、决策与整理
想法。它们不依赖具体领域或技术栈，产物是双方达成的共识而非文件，过程材料留
在对话里而不是你的工作区。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

默认按项目安装；若希望这套工作方式在所有会话中都可用，可加 `-g` 全局安装。

## Skill 列表

| Skill | 说明 |
|---|---|
| [clarify-thinking](clarify-thinking/) | 追问一个计划、想法或决定，直到 agent 与用户达成同一理解：将其建模为决策树，每轮把当前可问的问题连同推荐答案通过宿主自带的提问工具抛出，所有事实自己去查，且不在工作区留下任何文件。 |
