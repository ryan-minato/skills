## Why

A pull request carries up to three `spec/*` labels at once: the approval label, one of `spec/archived` | `spec/unarchived`, and one of `spec/not-started` | `spec/in-progress` | `spec/done`. On #122 the maintainer added `spec/approved` and expected the state to change, but `spec/unarchived` and `spec/not-started` stayed, because they belong to other axes. Three axes from two scripts read as conflicting states, and the progress axis repeats what `tasks.md` and `just spec-changes status` already show. hoshiori-dev/devcontainer-features keeps one specification-state label per pull request, decided by one script from the pull request's present state; this repository adopts that model.

## What Changes

- A pull request carries at most one `spec/*` label: `spec/pending` (it holds an unarchived change whose approval does not hold), `spec/approved` (the approval holds), or `spec/archived` (it holds no unarchived change and archives one). A pull request with neither carries none.
- `scripts/spec_approval.ts` decides and writes that one label. The approval rules (record, package comparison, withdrawal, fail closed) are unchanged, except that a person adding `spec/pending` after the latest adding of `spec/approved` also withdraws the approval. After an error, an open pull request gets `spec/pending` in place of `spec/approved`.
- `spec-labels.yml` runs that script alone; the snapshot step and the label-plan step go.
- `scripts/spec_changes.ts` loses its `labels` command and its label taxonomy; `status` keeps showing task progress locally.
- `.github/labels.json` declares `spec/pending`, `spec/approved`, `spec/archived` and drops `spec/unarchived`, `spec/not-started`, `spec/in-progress`, `spec/done`; `validate_harness.ts` checks the declaration against the taxonomy `spec_approval.ts` exports.
- The knowledge files, `ARCHITECTURE.md`, and `CONTRIBUTING.md` describe the one state label.

## Skills touched

repository change.

## Installed behavior

None: no skill changes. On GitHub, each pull request in this repository shows one specification state; task progress is read locally with `just spec-changes status`.

## Impact

- Files: `.github/workflows/spec-labels.yml`, `.github/labels.json`; `scripts/spec_approval.ts`, `spec_approval_test.ts`, `spec_changes.ts`, `spec_changes_test.ts`, `validate_harness.ts`, `validate_harness_test.ts`; `.agents/knowledge/spec-workflow.md`, `github-workflow.md`, `github-checks.md`; `ARCHITECTURE.md`; `.github/CONTRIBUTING.md`; `justfile` where it lists the `spec-changes` commands.
- Untouched: everything under `skills/` (the `openspec-workflow` and `spec-kit-workflow` skills keep delivering their multi-axis label assets), `openspec/specs/`, archived changes.
- GitHub: after merge the new labels are created by `labels / sync`; the four retired labels stay on the repository and on open pull requests until the maintainer prunes them (`sync_labels.ts --apply --prune`), which the workflow never does.
- Sequencing: #122 also edits `spec_changes.ts`, `spec-workflow.md ## Request automation`, `github-checks.md`, and `ARCHITECTURE.md`; this branch rebases onto `main` once #122 merges.

## Non-goals

- No change to the public skills' label assets or their specs.
- No change to what the approval package is, to the record comment's format, or to `checks / spec`.
- Removing the `snapshot` command, which loses its last workflow caller with this change and #122: a separate decision.

## Deliverable and pre-authorized actions

A pull request from `chore/single-spec-label`. Pre-authorized by the maintainer in conversation: pushes to that branch, the draft pull request and edits to its body.

## Tracked work

None; requested in conversation after #122's labels.
