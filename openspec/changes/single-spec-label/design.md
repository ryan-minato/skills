## Context

`spec-labels.yml` (`pull_request_target`, main's copy, `pull-requests: write`) runs three steps: `scripts/spec_approval.ts --apply` decides `spec/approved` from the label history, the record comment, and the approval package at the head; `spec_changes.ts snapshot` reads the head through the API; `spec_changes.ts labels` plans the archive axis and the progress axis from the related changes, and a shell loop applies the plan against a literal `case` list. `validate_harness.ts` (`spec`) checks `labels.json` against `MANAGED_LABELS` from `spec_changes.ts` and that `case` list. The reference model is `scripts/sync_pr_labels.ts` in hoshiori-dev/devcontainer-features: one state label of `pending | approved | archived`, or none, decided from the head tree and the file list. Binding constraint: nothing under `skills/` changes. See proposal.md for motivation.

## The state

Decided by `spec_approval.ts` from the pull request's present state, in this order:

| Head holds an unarchived change | Approval holds (rules unchanged) | Pull request adds a path under `openspec/changes/archive/` | State |
|---|---|---|---|
| yes | yes | — | `spec/approved` |
| yes | no | — | `spec/pending` |
| no | — | yes | `spec/archived` |
| no | — | no | none |
| no | — | file list cut short | `spec/archived` left as it is; the other two removed |

- *Unarchived change*: a directory directly under `openspec/changes/` other than `archive/`, read from the head tree the script already reads.
- *Adds a path*: a file of the pull request's file list (`pulls/<n>/files`) with status `added` or `renamed` under the archive directory. The list is cut short when it holds fewer entries than the pull request's `changed_files`, or that count exceeds 3000, the API's limit.
- One more withdrawal: a person adding `spec/pending` after the latest adding of `spec/approved` withdraws the approval, as a person removing `spec/approved` does.
- Writes: the record and withdrawn records first, as now; then the wanted state label is added, then every other state label on the pull request is removed. Labels outside the three are never touched.
- After an error, on an open pull request: add `spec/pending`, then remove `spec/approved`; on a closed one: remove `spec/approved`. The workflow's failure step keeps removing `spec/approved`.
- Read-only `just spec-approval <n>` prints the state, the reason, and the writes; exit 0 for `spec/approved`, 1 for any other state.

## Placement

| Item | File and section | Check that proves it |
|---|---|---|
| State decision and writes | `scripts/spec_approval.ts`: `STATE_LABELS`, `decide`, `writesOf`, `failClosed`, the `Api` label calls taking one of `STATE_LABELS`, a `files` read; header and usage | `just test` (`spec_approval_test.ts`: one case per table row, the `spec/pending` withdrawal, fail-closed on open and closed) |
| Workflow | `.github/workflows/spec-labels.yml`: the snapshot and plan steps removed; header | `spec_approval_test.ts` assertions on the file; `just validate` (`workflows`) |
| Taxonomy | `.github/labels.json`: three `spec/*` rows; `validate_harness.ts` `spec` check imports `STATE_LABELS`, drops the `case`-list check | `just validate`; `validate_harness_test.ts` |
| `labels` command removed | `scripts/spec_changes.ts` command table, parser, taxonomy constants, usage; `spec_changes_test.ts`; `justfile` comment | `just test`; `spec_changes.ts labels` exits 2 |
| Docs | `spec-workflow.md ## Approval gates`, `## Request automation`; `github-workflow.md` label table; `github-checks.md` `spec / labels` row; `ARCHITECTURE.md` approval section; `CONTRIBUTING.md` | `git grep -nE 'spec/(unarchived|not-started|in-progress|done)'` outside `skills/`, `openspec/specs/`, archives returns nothing |

## External impact

- `skills/**`, `openspec/specs/**`, archived changes: unchanged; `git diff --stat origin/main -- skills openspec/specs openspec/changes/archive` is empty.
- GitHub: `labels / sync` creates `spec/pending` after merge and corrects the descriptions; the four retired labels remain until the maintainer prunes them. The first `spec / labels` run on each open pull request replaces its state labels; retired labels on it stay until the prune.

## Decisions

- **One script, `spec_approval.ts`**: it already reads the head tree and owns the fail-closed path; the state needs one more read (the file list). Rejected: keeping `spec_changes.ts labels` for `spec/archived` (two writers of one label set reintroduce the problem); renaming the script (churn in the recipe, the deny-safe read path, and every document, for no behavior).
- **Head tree, not related changes**: `main` never holds an unarchived change (`checks / spec` fails the push), so the head's unarchived changes are the pull request's own; the approval rules already use the head tree.
- **Drop the progress axis**: `tasks.md` and `just spec-changes status` show it; no rule reads the labels.
- **`spec/archived` replaces `spec/approved` after the freeze**: nothing after the freeze reads the label; the record comment stays as the approval's evidence.
- **No migration code**: the prune is one maintainer command; a script that deletes retired labels would widen what the token writes.

## Risks / Trade-offs

- [An archive commit is reverted after the freeze] → `spec/approved` is no longer on the pull request, so the state is `spec/pending` and the record is withdrawn; the maintainer approves again. Fail closed, rare.
- [The pull request lists more than 3000 files] → `spec/archived` is left as it is; the reason says so.
- [Merge order with #122] → this branch rebases after #122 merges and resolves the overlapping edits.

## Verification plan

Script and tool harnesses:
- `just check` passes; `just test` covers every row of the state table, the `spec/pending` withdrawal, the write order, and both fail-closed paths.
- Read-only `./scripts/spec_approval.ts <n>` from this branch against #122 and against this pull request prints one state each, and the writes it would make, with no write.
- `./scripts/spec_changes.ts labels` exits 2; `just spec-changes status` and `just spec-check origin/main HEAD --draft` still work.
- The `git grep` in Placement returns nothing; `git diff --stat origin/main -- skills openspec/specs openspec/changes/archive` is empty.

Skipped:
- A live `spec / labels` run with the new code before merge: the workflow runs `main`'s copy, so the first live run is on the next pull request event after merge; the maintainer can dispatch nothing else to try it earlier.
