## 1. Skill and repository files

- [ ] 1.1 `scripts/validate_harness.py` mirror origin and docstring, `.agents/knowledge/harness-maintenance.md` origin row and slot-table row, `.github/workflows/spec-archive.yml` header comment; verify `just validate` and the mirror `diff` — proves Archive mirror
- [ ] 1.2 `.agents/knowledge/spec-workflow.md`: Lifecycle and tracked-work approval to discussion-closed, the Purpose-line exception, the mirror origin sentence; verify `just validate` — proves Specification gate, Purpose-line exception
- [ ] 1.3 `.agents/skills/change-workflow/SKILL.md` §3 and §7, `.github/PULL_REQUEST_TEMPLATE.md` approval line and checklist item 2, `.agents/knowledge/agent-authority.md` ready condition; verify `scripts/check_pr_policy.py` on a draft and a ready body — proves Specification gate
- [ ] 1.4 `.agents/skills/code-review/SKILL.md` contract-flow rule, `ARCHITECTURE.md` catalog description, `.claude-plugin/marketplace.json` `meta` description; verify `just gen-marketplace` changes nothing — proves Contract flow

## 3. Tests

- [ ] 3.1 Mirror and origin commands from the verification plan — proves Archive mirror
- [ ] 3.2 Clean-context readback of the four gate files and the two rule questions — proves Specification gate, Purpose-line exception, Contract flow
- [ ] 3.3 `check_pr_policy.py` draft-pass and ready-fail runs — proves Specification gate

## 4. Finish

- [ ] 4.1 Run `just check`, record the results in the pull request's Validation section, and archive this change inside the pull request together with `sdd-pilot-lessons`
