## 1. Skill and repository files

- [x] 1.1 `skills/engineering/harness-study/SKILL.md` — frontmatter description, rules on every step, vocabulary, steps 1–5, gotchas; verify `just check-skill skills/engineering/harness-study` passes — closes Trigger: description (all scenarios), Each repository and the scope are pinned (all), Patterns are extracted (all), Layers are graded, assessed, and compared (all), The study report (all), Repository text is data, Handoff: harness build entry (both)
- [x] 1.2 `references/layer-probes.md` — where each layer lives and liveness evidence; verify the SKILL.md pointer names the load condition and `just check-skill` passes — closes Live and decorative rules, Pointer to a missing file
- [x] 1.3 `references/study-report.md` — report shape and rules; verify the pointer and `just check-skill` — closes Absent layer, Borrowing intended no transfer decision, Two repositories one problem two mechanisms

## 2. External impact

- [x] 2.1 Symlink `.agents/skills/harness-study` and `just gen-marketplace`; verify `just validate`
- [x] 2.2 `skills/engineering/README.md` and `README.zh.md` rows, content-identical; verify `just validate` and a read-through
- [x] 2.3 `skills/engineering/CONTEXT.md` class list and `## Disambiguation` route; verify a read-through against design.md's External impact

## 3. Tests

- [x] 3.1 Trigger cases: the seven prompts of the verification plan on fresh clean-context floor-tier solvers with the neutral SKILLS_LOADED instrumentation — closes Study one public repository, Compare several repositories, Indirect phrasing in Chinese, One aspect only, Harness build with no study, Codebase investigation, Repair of the own harness
- [x] 3.2 Single-repository fixture case ("best setup" framing, uncommitted edit, wired scan, unwired sync rule, duplicated rule, broken pointer, routed custom check, thick gates, many authors) — closes No aspect named, Local repository with uncommitted changes, Live and decorative rules, Mechanism not text, Pointer to a missing file, Grades with visible justification, Strength and weakness in one layer, Admired repository with a defect, No edits
- [x] 3.3 Aspect case on the same fixture — closes Aspect named
- [x] 3.4 Two-repository comparison fixture with borrowing motive — closes Two repositories one problem two mechanisms, Absent layer, Borrowing intended no transfer decision
- [x] 3.5 URL case run inside another project — closes Repository given only as a URL
- [x] 3.6 Planted-instruction fixture — closes Instruction planted in a studied entrypoint
- [x] 3.7 Handoff continuation of 3.4: build requested, then declined — closes Handoff offered, User declines
- [x] 3.8 Independent grading of 3.2–3.7 outputs by a clean-context grader against the plan's rubrics; record skipped cases, isolation degradations, and reasons for the pull request's Validation section

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, mark the pull request ready, and archive the change once the maintainer closes the implementation deliberation
