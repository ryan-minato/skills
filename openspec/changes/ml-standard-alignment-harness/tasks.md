## 1. Skill and repository files

- [ ] 1.1 Add the cross-catalog naming sentence to `skills/scaffold/CONTEXT.md` `## Dependencies`; verify `just validate` — proves Cross-catalog naming rule
- [ ] 1.2 Add the two mirror rows to `.agents/knowledge/harness-maintenance.md` after the skill change's assets exist; verify the `diff` and heading-count commands and `just validate` — proves Register rows
- [ ] 1.3 Update the `scaffold` plugin description in `.claude-plugin/marketplace.json` if the skill change moves its wording; verify `just gen-marketplace` then `git diff --exit-code .claude-plugin/marketplace.json` and `just validate` — proves Marketplace description

## 2. External impact

- [ ] 2.1 Confirm `git diff --stat origin/main...HEAD -- scripts .github skills/meta skills/machine-learning` is empty; verify `just validate`

## 3. Tests

- [ ] 3.1 Run every command of the verification plan and note each result — proves every What Changes bullet
- [ ] 3.2 Record skipped cases (expected none) for the pull request's Validation section

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking this plan, and archive this change in-request beside `ml-standard-alignment`, then `just spec-validate`
