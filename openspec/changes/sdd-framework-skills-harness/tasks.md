## 1. Skill and repository files

- [ ] 1.1 Catalog scaffold: `skills/sdd/{README.md,README.zh.md,CONTEXT.md}`, `ARCHITECTURE.md` `## Catalogs`, root `README.md` + `README.zh.md`, `.github/labels.json` `catalog/sdd`, the three issue forms' Catalog dropdown; verify `just validate`
- [ ] 1.2 Symlinks: retarget `.agents/skills/spec-driven-development`, add `openspec-workflow` and `spec-kit-workflow`; `just gen-marketplace`; verify `just validate` and `just spec-sync` leaves the symlinks untouched
- [ ] 1.3 Workflows: replace `.github/workflows/spec-archive.yml`, add `spec-command.yml` and `spec-labels.yml`, edit `checks.yml` (activity types; `checks / spec` steps); verify YAML parse, `grep -rn '{{[A-Z]' .github/` empty
- [ ] 1.4 Script and recipes: `scripts/spec_changes.py` mirror, delete `scripts/archive_completed_changes.py`, `justfile` `spec-check` and `spec-changes`; verify the mirror diff, `just spec-changes --help`, `just lint`
- [ ] 1.5 Labels and validator: six `spec/*` rows in `.github/labels.json`; `scripts/validate_harness.py` mirror pair, `workflow` applier, `check_spec_labels()`, scan set; verify `just validate` and the `sync_labels.py` dry run
- [ ] 1.6 Contract and project skill: `.agents/knowledge/spec-workflow.md`, `.agents/skills/change-workflow/SKILL.md`; verify a clean-context readback
- [ ] 1.7 Templates and schema: `.github/PULL_REQUEST_TEMPLATE.md`, `openspec/config.yaml`, `openspec/schemas/skill-change/schema.yaml`; verify `just validate`, `check_pr_policy.py` over a template-built body, `just spec-validate`
- [ ] 1.8 Knowledge and maps: `AGENTS.md`, `ARCHITECTURE.md`, `.agents/knowledge/{agent-authority,github-workflow,github-checks,github-settings,harness-maintenance}.md`, `skills/machine-learning/CONTEXT.md`; verify `just validate` and the residue grep

## 2. External impact

- [ ] 2.1 `skills/engineering/{README.md,README.zh.md,CONTEXT.md}` without the moved skill; verify `just validate` and a read of the pair
- [ ] 2.2 `scripts/validate_skills.py` untouched; verify `git diff --stat origin/main...HEAD -- scripts/validate_skills.py` empty

## 3. Tests

- [ ] 3.1 Catalog scaffold proofs (validate, marketplace diff, symlink listing, README pair reads, spec-sync)
- [ ] 3.2 Automation proofs (YAML parse, pinned SHAs, no push on the fork branch, `just spec-check` draft and ready verdicts, mirror diff, dry-run label list, the `spec/extra` negative case)
- [ ] 3.3 Contract, project skill, template, and schema proofs (clean-context readback, `check_pr_policy.py` runs, `just spec-validate`)
- [ ] 3.4 Record skipped cases (live bot runs after merge) for the pull request's Validation section

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section, archive this change inside the pull request by hand
