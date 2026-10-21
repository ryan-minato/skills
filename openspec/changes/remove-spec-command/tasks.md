## 1. Repository files

- [x] 1.1 Remove `.github/workflows/spec-command.yml`; drop the `spec / command` row and healthy-run line from `github-checks.md` and its entry from the privileged-workflow line; verify `just validate`
- [x] 1.2 Remove the `show` mode of `scripts/spec_changes.ts` (`--url-prefix`, `--max-chars`, `showMarkdown`, `cmdShow`, usage text) and its tests, and the header lines naming `spec / command`; verify `just test` and that `./scripts/spec_changes.ts show` exits 2
- [x] 1.3 Remove the `/spec` mentions from `spec-workflow.md`, `agent-authority.md`, `harness-maintenance.md`, `ARCHITECTURE.md`, and the `change-workflow` skill; verify `just validate` and `just check-skill .agents/skills/change-workflow`

## 2. Verification

- [ ] 2.1 Run the verification plan of `design.md`: `just check`; the `git grep` for `spec / command`, `spec-command`, and `/spec` commands outside `skills/`, `openspec/specs/`, and archives returns nothing; no `issue_comment` under `.github/`; no diff under `skills/`, `openspec/specs/`, or archived changes
