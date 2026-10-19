## 1. Skill and repository files

- [ ] 1.1 `skills/sdd/spec-driven-development/SKILL.md`: step 1 asks for the four answers; `## Specification quality` gains the four-answers rule; new `## Artifact roles` section with the two tests and the two-list acceptance; the approval package reads the acceptance lists first; verify `just check-skill skills/sdd/spec-driven-development` — closes A specification answers done, non-goals, deliverable, and pre-authorized actions; Each record artifact owns one role, tested by placement rules
- [ ] 1.2 `skills/meta/meta-spec-workflow/SKILL.md`: question 1 (spec-as-source comparison check), question 3 (recorded-approval implementations offered; the pre-authorized-actions line), question 4 (artifact roles, tests, acceptance shape); verify `just check-skill skills/meta/meta-spec-workflow` — closes the four meta-spec-workflow Behavior requirements
- [ ] 1.3 `references/contract-design.md`: `## Artifact roles`, `## Specification as source for a surface`, the implementations table under Approval modes; verify `just check-skill` — closes the same
- [ ] 1.4 `references/durable-output.md`: must-carry bullets for roles, pre-authorized actions, and the comparison check; survival rules for dated observations and bootstrap narrative; verify `just check-skill` — closes Observations are dated and bootstrap narrative is excluded
- [ ] 1.5 `assets/spec-workflow.md`: `## Artifact roles` section, the source-of-truth row for a ruled surface, the transient-exception trigger; verify `just check-skill` — closes the deposited-contract requirement

## 2. External impact

- [ ] 2.1 `skills/sdd/README.md` and `README.zh.md` rows; `skills/meta/README.md` and `README.zh.md` rows; verify `just validate`
- [ ] 2.2 Read-through of `skills/sdd/CONTEXT.md`, `skills/meta/CONTEXT.md`, `openspec-workflow`, and `spec-kit-workflow` for contradictions; no edit expected

## 3. Tests

- [ ] 3.1 Outcome cases for `spec-driven-development` (three cases) per the verification plan — closes its Behavior scenarios
- [ ] 3.2 Outcome cases for `meta-spec-workflow` (four cases) per the verification plan — closes its Behavior scenarios
- [ ] 3.3 Record skipped cases and their reasons for the pull request's Validation section

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and leave the change unarchived for the maintainer's second deliberation
