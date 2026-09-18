## 1. Skill and repository files

- [x] 1.1 Add `.github/workflows/labels-sync.yml` (job `sync` named `labels / sync`: path-filtered push to `main`, weekly schedule, manual dispatch; `permissions: {}` with job-level `contents: read` and `issues: write`; pinned checkout; `refs/heads/main` guard; `sync_labels.py --apply` without `--prune`; plan, summary, and one warning per prune candidate) and the `labels / sync` row of `.agents/knowledge/github-checks.md` in the same commit; verify `just validate` passes — proves The workflow, Checks table row
- [x] 1.2 Update the Labels row of `.agents/knowledge/github-settings.md`, the "a catalog added or removed" row of `.agents/knowledge/harness-maintenance.md`, and `ARCHITECTURE.md` `## GitHub Workflow`; verify `just validate` passes — proves Settings register, Maintenance register, Architecture map

## 2. External impact

- [x] 2.1 Confirm `git diff --stat origin/main...HEAD -- scripts .github/labels.json skills` is empty and record the post-merge dispatch and readback as the maintainer's step in the pull request handover

## 3. Tests

- [x] 3.1 Run every command of the verification plan: the workflow read-through, the summary shell against a real dry-run plan and a hand-made plan with two prune candidates and one update (never `--apply`), `just validate` with and without the checks-table row in a scratch copy, and the register read-throughs; note each result — proves every What Changes bullet
- [x] 3.2 Record skipped cases and their reasons for the pull request's Validation section (expected: the post-merge dispatch, which runs only on `main`)

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and archive this change in the pull request once the maintainer closes the implementation deliberation, then `just spec-validate`
