# engineering

[English](README.md)

通用编程**方法论**类 skill——跨语言、跨框架适用的方法、工作流和实践——
外加不足以自立 catalog 的窄域**工件创作**工作流（如 Dev Container 工件与
持久化视觉设计规范）。构建 GitHub 或 GitLab 项目的完整生命周期
harness——包括协作文件、规范与日常平台工作流——属于一次性的 `meta`
catalog。

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

**brownfield 套件**——五个以 `brownfield-` 为前缀的 skill——面向既存代码库：
以证据理解代码库、为新工程师做入职引导、判定哪些行为是契约并为其加上护栏，
以及在重写中保持边界行为。套件成员相互依赖，请一起安装：

```bash
npx skills add ryan-minato/skills \
  --skill brownfield-intelligence --skill brownfield-investigation \
  --skill brownfield-onboarding --skill brownfield-specification \
  --skill brownfield-migration
```

## Skill 列表

| Skill | 说明 |
|---|---|
| [brownfield-investigation](brownfield-investigation/) | 以证据调查既存代码库：按问题选取所需视角（文档与代码核对、仓库地图、领域模型、运行时流程、数据与状态、候选契约、测试安全网、历史）与深度（ORIENT、ESTABLISH、EXHAUSTIVE），并固定 revision；每条发现都记录来源、置信度、反证与未知项；报告文档漂移与疑似 bug，但不编辑、不修复任何内容；宿主支持时把相互独立的视角并行交给子代理，否则顺序执行。 |
| [brownfield-onboarding](brownfield-onboarding/) | 为加入既存代码库的工程师编写入职材料：构建一个最小充分、可信的心智模型（目的、如何运行、组件、术语、架构、代表性流程、改什么去哪里、风险与未知项），以调查得到的证据为基础，复用经核实的文档，把观察到的惯例描述为当前做法而非规则，并验证给出的每个路径与命令。 |
| [code-refactoring](code-refactoring/) | 以测试保障的小步、保持行为不变的方式重构既有代码：把结构调整与行为变更分离，判断何时重构（何时不重构），诊断代码坏味道，并安全地执行标准的具名重构手法。 |
| [devcontainer-authoring](devcontainer-authoring/) | 创作、测试与发布 Dev Container 工件——Feature（install.sh 契约、幂等性与多基础镜像质量标准、独立性规则）、Template（选项替换、载荷设计、冒烟测试循环）与预构建镜像（devcontainer build --push、metadata 合并语义）——附带仓库脚手架与共享 action CI。 |
| [design-md](design-md/) | 创作并校验持久化、供 agent 读取的 DESIGN.md 视觉设计规范，包含可选 YAML 设计 token、正文指导、上游格式检查和 OKLCH 计算器。 |
| [gitmoji](gitmoji/) | 起草 gitmoji 提交信息：先确定项目变体（独立语法 vs 叠加 CC 语法、unicode vs 文本代码），再通过首个匹配即停的决策列表为主要意图选出唯一 emoji，最后按交付前清单校验。 |
| [goal-alignment](goal-alignment/) | 与用户对齐创建物（软件、系统、实验、skill、服务等）应达成的目标：以一轮轮追问推进直至共识（可推断处附建议答案，仅用户可知的事实则直接提问），再把共识记录为单一事实来源的目标文档——整体目标、带验证方式与层级（硬约束/优化目标/偏好）的具体目标、分级要求（强制/尽力/偏好）与权衡决策记录。只谈目标；不含计划与架构。 |
| [knowledge-deposition](knowledge-deposition/) | 把一条已确认的知识沉淀进项目，让未来的 agent 能找到并遵循：探测项目已有的 agent 指导存放位置，选择合适的载体（entrypoint 行仅用于每个会话都必须看到的内容，知识库文件加事件指针为默认，项目 skill 用于复现且脆弱的过程，或先搁置等待复现），写成可独立执行的指令，并注册事件触发式指针。 |
| [session-retrospective](session-retrospective/) | 把一次工作会话蒸馏为持久的项目经验：从对话中挖掘六类信号（反复失败、工具反复标记的修正、代价高昂的发现、实验结论、被用户否决的默认做法、文档与实际不符），以"复现性乘以影响"权衡记录带来的上下文租金，并呈现按序排列的 findings 清单供逐条批准——只产出 findings，用户批准前不写任何内容。 |
