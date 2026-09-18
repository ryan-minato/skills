## 1. Catalog scaffold and the moved skill's harness

- [x] 1.1 `skills/sdd/{README.md,README.zh.md,CONTEXT.md}`, `ARCHITECTURE.md` `## Catalogs`, root `README.md` + `README.zh.md`, `.github/labels.json` `catalog/sdd`, the three issue forms' Catalog dropdown; verify `just validate`
- [x] 1.2 `.agents/skills/` symlinks for the three `sdd` skills; `.claude-plugin/marketplace.json`; `skills/engineering/{README.md,README.zh.md,CONTEXT.md}` without the moved skill; `skills/machine-learning/CONTEXT.md` pointers; verify `just validate`, `just gen-marketplace` with no drift, and `just spec-sync` leaving the symlinks in place
- [x] 1.3 `scripts/validate_skills.py` untouched; verify `git diff --stat origin/main...HEAD -- scripts/validate_skills.py` empty

## 2. Request automation

- [x] 2.1 `.github/workflows/`: `spec-command.yml` and `spec-labels.yml` filled from the skill's assets — base checkout only, the head read through `scripts/spec_changes.py snapshot`, the command restricted to the collaborator associations, the platform token on the steps that use it, the label plan checked against the literal taxonomy; `spec-archive.yml` deleted; `checks.yml` gains the `ready_for_review` and `converted_to_draft` activity types and its `checks / spec` steps; verify the files parse, `grep -rn 'git push\|git .*fetch\|actions/checkout' .github/workflows/spec-*.yml` finds only base checkouts, and `grep -rn '{{[A-Z]' .github/` is empty
- [x] 2.2 `scripts/spec_changes.py` mirrored from the skill byte for byte; `scripts/archive_completed_changes.py` deleted; `justfile` recipes `spec-check` and `spec-changes` replace `spec-archive-completed`; verify the mirror diff empty, `just spec-changes --help`, and `just lint`
- [x] 2.3 `.github/labels.json` carries five `spec/*` rows; `scripts/validate_harness.py` carries the mirror pair, the `workflow` applier restricted to `spec/`, `check_spec_labels()` against the script's taxonomy, and the hard-coded-version scan; verify `just validate`, the `sync_labels.py` dry run listing `catalog/sdd` and the five labels, and a scratch `spec/extra` row making `just validate` fail

## 3. Contract, project skill, templates, knowledge

- [x] 3.1 `.agents/knowledge/spec-workflow.md`, `.agents/skills/change-workflow/SKILL.md`, `AGENTS.md`, and `.github/PULL_REQUEST_TEMPLATE.md` describe one order: the draft opens with the complete approval package, the request is marked ready for the deliberation on the finished implementation, the implementer archives once that closes, approval applies to the frozen version, and `checks / spec` is red until the freeze; verify a read-through and `python3 scripts/check_pr_policy.py` over a body built from the template
- [x] 3.2 `openspec/config.yaml` and `openspec/schemas/skill-change/schema.yaml` carry the design instruction (part of the approval package, bounds not steps, no secrets); verify `just spec-validate`
- [x] 3.3 `.agents/knowledge/{github-checks,github-settings,harness-maintenance,agent-authority,github-workflow}.md` and `ARCHITECTURE.md` carry the two `spec / *` job rows, the privileged-read rule, the mirror rows, the label rows, and no bypass requirement; verify `just validate` and a read-through

## 4. Proofs

- [x] 4.1 Catalog scaffold proofs (validate, marketplace drift, symlink listing, README pair reads, spec-sync)
- [x] 4.2 Automation proofs (files parse, pinned SHAs, no push and no head read in any `spec / *` workflow, `just spec-check` draft and ready verdicts, mirror diff, dry-run label list, the `spec/extra` negative case)
- [x] 4.3 Contract and project skill proofs (a clean-context read states the draft's first content, when the request is marked ready, who archives and when, and what the archive freezes)
- [x] 4.4 This request runs the order it installs: ready for the deliberation with `checks / spec` red, archived once the maintainer closes it, green after

## 5. Finish

- [x] 5.1 Run `just check`, write the results to the pull request's Validation section, and archive this change inside the pull request after the deliberation closes
