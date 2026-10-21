## 1. The state label

- [x] 1.1 `scripts/spec_approval.ts`: `STATE_LABELS`, the state decision of design.md `## The state label` (head tree, file list, the `spec/pending` withdrawal), writes that add the wanted state and remove the others, fail closed with `spec/pending` on an open pull request; header, usage, and the record's wording
- [x] 1.2 `scripts/spec_approval_test.ts`: one case per row of the state table, a package change after the recorded commit, a `tasks.md`-only change and an archive commit, the `spec/pending` withdrawal, write order, both fail-closed paths, the file-list read

## 2. The labels workflow and the taxonomy

- [x] 2.1 `.github/workflows/spec-labels.yml`: the snapshot and plan steps removed, header rewritten; its assertions in `spec_approval_test.ts` follow
- [x] 2.2 `.github/labels.json`: `spec/pending`, `spec/approved`, `spec/archived` only
- [x] 2.3 `scripts/validate_harness.ts` `spec` check against `STATE_LABELS`, without the `case`-list check; `validate_harness_test.ts`

## 3. `spec_changes.ts`

- [x] 3.1 Remove the `labels` command and its taxonomy constants; `check` with a request source warns on an unarchived change and `--draft` is gone; usage, header, and `justfile` comments
- [x] 3.2 `scripts/spec_changes_test.ts` follows

## 4. The archive gate

- [x] 4.1 `.github/workflows/checks.yml`: `checks / archive-verdict` and the carrier job, the default-branch condition, `checks / spec` without `--draft`
- [x] 4.2 `scripts/validate_harness.ts` `checks` check reads a conditional job name as its two names; `validate_harness_test.ts`

## 5. Documents

- [x] 5.1 `github-checks.md` (table, healthy runs, the no-standing-failure rule, diagnosing), `github-settings.md` required checks
- [x] 5.2 `spec-workflow.md`, `github-workflow.md`, `agent-authority.md`, `ARCHITECTURE.md`, `.github/CONTRIBUTING.md`, `change-workflow` skill

## 6. Verification

- [ ] 6.1 `just check`; the read-only `spec_approval.ts` runs against #122 and #123; `labels` and `check --draft` exit 2; the `git grep` of design.md Placement; no diff under `skills/`, `openspec/specs/`, archives
- [ ] 6.2 On #123: `checks / archive-verdict`, `checks / awaiting-archive`, `checks / spec`, `checks / gate` green and no `checks / archived`; a throwaway draft pull request into `chore/single-spec-label` reports `checks / archived`, then is closed and its branch deleted
