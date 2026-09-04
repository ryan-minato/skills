## 1. Skill and repository files

- [ ] 1.1 Fix the spec layout convention and the repository-change rule in `.agents/knowledge/spec-workflow.md`, `openspec/config.yaml`, `ARCHITECTURE.md`, `AGENTS.md`, and `change-workflow` §3.4; verify `just validate` — proves "Spec layout convention"
- [ ] 1.2 Land the `skill-change` schema as default with the validator check; verify `openspec schemas --json` and `python3 scripts/validate_harness.py` — proves "Project schema default"
- [ ] 1.3 Add `scripts/archive_completed_changes.py` as a mirror of the SDD skill's script (pair registered in `scripts/validate_harness.py`) and the `just spec-archive-completed` recipe; verify the script harness of the verification plan — proves "`spec-archive` workflow and script" (script half)
- [ ] 1.4 Add `.github/workflows/spec-archive.yml` and list its job in `github-checks.md`, the bypass action in `github-settings.md`, the trigger in `harness-maintenance.md`; verify the workflow read-through and `python3 scripts/validate_harness.py` — proves "`spec-archive` workflow and script" (workflow half)
- [ ] 1.5 Write the combined shape and automated archive mode into `spec-workflow.md` (shape, archive mode, lifecycle), `agent-authority.md` ready conditions, `openspec/config.yaml` archive guidance, and the companion-change naming into `spec-workflow.md` and `github-workflow.md`; verify readback — proves "Combined shape, automated archive", "Companion change naming"
- [ ] 1.6 Rework `.agents/skills/change-workflow/SKILL.md` (draft after clarify, approval comment, plan and tasks after approval, ready without archiving, gotchas) and `.github/PULL_REQUEST_TEMPLATE.md` (two specification items); verify `scripts/check_pr_policy.py` on a body built from the template — proves "Combined shape, automated archive"
- [ ] 1.7 Put the verification plan in `design.md` through `skill-authoring/SKILL.md`, `references/testing.md`, and the template's Validation comment; verify readback — proves "Verification plan in design.md"

## 2. External impact

- [ ] 2.1 Run `just validate` and `git grep -n 'openspec/specs/<domain>\|repository/<tool>\|Spec: none'` to confirm no stale layout rule remains

## 3. Tests

- [ ] 3.1 Run the script harness (`--help`, `--dry-run`, fixture archive in a temporary worktree, repeated run, `--bogus`) — proves the script bullet
- [ ] 3.2 Run `scripts/check_pr_policy.py` against a body from the new template, ticked and unticked — proves the template bullet
- [ ] 3.3 Run the clean-context readback of the changed knowledge files and skills — proves the knowledge bullets

## 4. Finish

- [ ] 4.1 Run `just check`, record results in the pull request's Validation section, and archive this change in-request together with `sdd-workflow-integration`, then `just spec-validate`
