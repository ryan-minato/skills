## 1. Skill and repository files

- [x] 1.1 Add the cross-catalog naming sentence to `skills/scaffold/CONTEXT.md` `## Dependencies`; verify `just validate` — proves Cross-catalog naming rule
- [x] 1.2 Add the two mirror rows to `.agents/knowledge/harness-maintenance.md` after the skill change's assets exist; verify the `diff` and heading-count commands and `just validate` — proves Register rows
- [x] 1.3 Update the `scaffold` plugin description in `.claude-plugin/marketplace.json` if the skill change moves its wording; verify `just gen-marketplace` then `git diff --exit-code .claude-plugin/marketplace.json` and `just validate` — proves Marketplace description
- [x] 1.4 Narrow the research-spec heading row to the two remaining files once the scaffold's separate skeleton is removed; verify the heading-list diff and `just validate` — proves Register rows

## 2. External impact

- [x] 2.1 Confirm `git diff --stat origin/main...HEAD -- scripts .github skills/meta` is empty and the `skills/machine-learning` diff holds only what the two skill changes on this branch carry; verify `just validate`

## 3. Tests

- [x] 3.1 Run every command of the verification plan and note each result — proves every What Changes bullet
- [x] 3.2 Record skipped cases (expected none) for the pull request's Validation section

## 4. Finish

- [x] 4.1 Run `just check`, write the results to the pull request's Validation section linking this plan, then `just spec-validate`; archiving waits for the maintainer's instruction
