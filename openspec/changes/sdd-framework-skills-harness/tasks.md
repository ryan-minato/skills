## 1. Skill and repository files

- [x] 1.1 Catalog scaffold: `skills/sdd/{README.md,README.zh.md,CONTEXT.md}`, `ARCHITECTURE.md` `## Catalogs`, root `README.md` + `README.zh.md`, `.github/labels.json` `catalog/sdd`, the three issue forms' Catalog dropdown; verify `just validate`
- [x] 1.2 Symlinks: retarget `.agents/skills/spec-driven-development`, add `openspec-workflow` and `spec-kit-workflow`; `just gen-marketplace`; verify `just validate` and `just spec-sync` leaves the symlinks untouched
- [x] 1.3 Workflows: replace `.github/workflows/spec-archive.yml`, add `spec-command.yml` and `spec-labels.yml`, edit `checks.yml` (activity types; `checks / spec` steps); verify YAML parse, `grep -rn '{{[A-Z]' .github/` empty
- [x] 1.4 Script and recipes: `scripts/spec_changes.py` mirror, delete `scripts/archive_completed_changes.py`, `justfile` `spec-check` and `spec-changes`; verify the mirror diff, `just spec-changes --help`, `just lint`
- [x] 1.5 Labels and validator: six `spec/*` rows in `.github/labels.json`; `scripts/validate_harness.py` mirror pair, `workflow` applier, `check_spec_labels()`, scan set; verify `just validate` and the `sync_labels.py` dry run
- [x] 1.6 Contract and project skill: `.agents/knowledge/spec-workflow.md`, `.agents/skills/change-workflow/SKILL.md`; verify a clean-context readback
- [x] 1.7 Templates and schema: `.github/PULL_REQUEST_TEMPLATE.md`, `openspec/config.yaml`, `openspec/schemas/skill-change/schema.yaml`; verify `just validate`, `check_pr_policy.py` over a template-built body, `just spec-validate`
- [x] 1.8 Knowledge and maps: `AGENTS.md`, `ARCHITECTURE.md`, `.agents/knowledge/{agent-authority,github-workflow,github-checks,github-settings,harness-maintenance}.md`, `skills/machine-learning/CONTEXT.md`; verify `just validate` and the residue grep

## 2. External impact

- [x] 2.1 `skills/engineering/{README.md,README.zh.md,CONTEXT.md}` without the moved skill; verify `just validate` and a read of the pair
- [x] 2.2 `scripts/validate_skills.py` untouched; verify `git diff --stat origin/main...HEAD -- scripts/validate_skills.py` empty

## 3. Tests

- [x] 3.1 Catalog scaffold proofs (validate, marketplace diff, symlink listing, README pair reads, spec-sync)
- [x] 3.2 Automation proofs (YAML parse, pinned SHAs, no push on the fork branch, `just spec-check` draft and ready verdicts, mirror diff, dry-run label list, the `spec/extra` negative case)
- [x] 3.3 Contract, project skill, template, and schema proofs (clean-context readback, `check_pr_policy.py` runs, `just spec-validate`)
- [x] 3.4 Record skipped cases (live bot runs after merge) for the pull request's Validation section

## 4. The privileged read path

- [x] 4.1 `.github/workflows/spec-labels.yml` and `spec-command.yml` regenerated from the updated assets: base checkout only, the head read through `scripts/spec_changes.py snapshot`, the label plan checked against the literal taxonomy; verify YAML parse, job names unchanged, and `grep -n 'git .*fetch'` empty in both
- [x] 4.2 `.github/workflows/spec-archive.yml` regenerated: the fork branch reads the snapshot, the same-repository steps carry the literal `head.repo.full_name == github.repository` condition, the `SAME_REPO` and `OPEN` environment variables are gone; verify YAML parse and `just validate` (`check_spec_labels`)
- [x] 4.3 `scripts/spec_changes.py` mirror refreshed; verify `diff scripts/spec_changes.py skills/sdd/openspec-workflow/scripts/spec_changes.py` empty and `just validate` green
- [x] 4.4 `.agents/knowledge/github-checks.md` records the rule, the snapshot in the two job rows, and the standing instruction never to add a head checkout or fetch; verify a read-through and `just validate`
- [x] 4.5 The open CodeQL alert on `spec-labels.yml` closes on the next default-setup scan of the branch; record the outcome in the pull request's Validation section

## 5. Workflow and knowledge corrections

- [x] 5.1 This repository's `spec-command.yml` grants `pull-requests: read` and `spec-archive.yml` posts fork instructions that fetch from `upstream`, both regenerated from the corrected assets; `scripts/spec_changes.py` mirror refreshed — verify YAML parse, `just validate`, and the mirror diff empty

## 6. This repository follows the skills

- [ ] 6.1 Delete `.github/workflows/spec-archive.yml` and the `spec/archive` row from `.github/labels.json`; rework `check_spec_labels()` and drop `SPEC_ARCHIVE_WORKFLOW` from `scripts/validate_harness.py`; drop the trigger label from `scripts/spec_changes.py` and its taxonomy output; remove the `spec / archive` row and the approval-click note from `.agents/knowledge/github-checks.md` — verify `just validate` and the mirror diff empty
- [ ] 6.2 Move the archive after the deliberation in `.agents/knowledge/spec-workflow.md`, the `change-workflow` project skill, `AGENTS.md`, and the pull request template's checklist; state that the required check stays red until the freeze — verify a read-through and `python3 scripts/check_pr_policy.py`
- [ ] 6.3 Regenerate `.github/workflows/spec-command.yml` and `spec-labels.yml` from the hardened assets, and refresh the `scripts/spec_changes.py` mirror — verify both parse, job names unchanged, and `diff` against the skill copy empty
- [ ] 6.4 Run this request through the new order: ready for the deliberation with the check red, archive after the maintainer closes it, then green

## 7. Finish

- [x] 7.1 Run `just check`, write the results to the pull request's Validation section, archive this change inside the pull request by hand
