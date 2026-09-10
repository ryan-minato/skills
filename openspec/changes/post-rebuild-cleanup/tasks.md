## 1. Skill and repository files

- [ ] 1.1 Drop `denoland.vscode-deno` from `.devcontainer/devcontainer.json`; verify `grep -c deno` prints 0 and `just validate` passes — proves "Dev container"
- [ ] 1.2 Delete `.agents/mcp_config.json`, drop its `ARCHITECTURE.md` row, rewrite the MCP row in `harness-maintenance.md` to three declarations; verify `just validate` and the `grep -rln agentskills.io` listing — proves "MCP declaration"
- [ ] 1.3 Update `github-settings.md`: bypass row records the API refusal and the two candidate paths, Last verification records the 2026-09-10 readback; verify against a fresh readback — proves "Settings register"
- [ ] 1.4 Unify the specification-boundary wording in `spec-workflow.md`, `ARCHITECTURE.md`, `schema.yaml`, the proposal and design templates, `AGENTS.md`, and `bug-report.yml`; verify the `git grep` prints nothing and `just spec-validate` passes — proves "Specification boundary wording"
- [ ] 1.5 Rewrite `.github/PULL_REQUEST_TEMPLATE.md` to the settled shape and extend `scripts/check_pr_policy.py` (ready-state rules, docstring) and `scripts/validate_harness.py` (template roles); verify the fixture dry runs (a)–(e) and `just validate` — proves "Pull request body shape" (template and checks)
- [ ] 1.6 Update `change-workflow` §5–§7, `spec-workflow.md` Specifications and tracked work, `agent-authority.md` report, and add the template row to `harness-maintenance.md`; verify the clean-context readback — proves "Pull request body shape" (skill and knowledge)
- [ ] 1.7 Replace `Specification approved at <sha>` with `Specification approved` and the timeline rule in `spec-workflow.md` Lifecycle, `change-workflow` §3, and the template; verify `git grep 'approved at'` prints nothing — proves "Approval record"

## 2. External impact

- [ ] 2.1 Run `just validate` and confirm `git status --short` lists only the files in the placement table and this change record
- [ ] 2.2 Delete `feat/project-code-review` and `feat/harden-github-harness` on `origin` (authorized; `refs/pull/69/head` keeps the unmerged commits); verify `git branch -r --merged origin/main | grep -v main` prints nothing — proves "Remote branches"

## 3. Tests

- [ ] 3.1 Run the fixture dry runs (a)–(e) of `check_pr_policy.py` — proves "Pull request body shape"
- [ ] 3.2 Run the clean-context readbacks (boundary question; `change-workflow` §6; `spec-workflow.md` Lifecycle step 2) — proves "Specification boundary wording", "Pull request body shape", "Approval record"
- [ ] 3.3 Run the readback commands of the settings register — proves "Settings register"
- [ ] 3.4 Record skipped cases and their reasons for the pull request's Validation section (expected: none)

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking this plan, fill the Changes section with permalinks, archive this change in-request (bypass not granted), run `just spec-validate`, and point the `Spec:` link at the archive path
