## Why

Two signals on a pull request misreport where its change stands.

- A pull request carries up to three `spec/*` labels at once: the approval label, one of `spec/archived` | `spec/unarchived`, and one of `spec/not-started` | `spec/in-progress` | `spec/done`. On #122 the maintainer added `spec/approved` and expected the state to change, but `spec/unarchived` and `spec/not-started` stayed, because they belong to other axes. Three axes from two scripts read as conflicting states, and the progress axis repeats what `tasks.md` and `just spec-changes status` already show.
- `checks / spec` fails a ready pull request that holds an unarchived change, so `checks / gate` is red through the whole second deliberation. Waiting for the freeze is the expected state, not a defect, yet it looks like one and hides real failures behind the same red mark.

hoshiori-dev/devcontainer-features keeps one specification-state label per pull request, decided by one script, and blocks the merge of an unarchived change by withholding a required check instead of failing one. This repository adopts both.

## What Changes

- A pull request carries at most one `spec/*` label: `spec/pending` (it holds an unarchived change whose approval does not hold), `spec/approved` (the approval holds), or `spec/archived` (it holds no unarchived change and archives one). A pull request with neither carries none.
- `scripts/spec_approval.ts` decides and writes that one label. The approval rules (record, package comparison, withdrawal, fail closed) are unchanged: a change to the approval package after the approved commit turns `spec/approved` back into `spec/pending` and withdraws the record, until the maintainer approves again. New: a person adding `spec/pending` after the latest adding of `spec/approved` also withdraws the approval. After an error, an open pull request gets `spec/pending` in place of `spec/approved`.
- `spec-labels.yml` runs that script alone; the snapshot step and the label-plan step go. `scripts/spec_changes.ts` loses its `labels` command and its label taxonomy.
- `.github/labels.json` declares `spec/pending`, `spec/approved`, `spec/archived` and drops `spec/unarchived`, `spec/not-started`, `spec/in-progress`, `spec/done`.
- New archive gate in `checks.yml`: a verdict job runs the base's `spec_changes.ts status` over the pull request's commits, and a carrier job succeeds under the name `checks / archived` when no related change is unarchived, or under `checks / awaiting-archive` while one is. The ruleset requires `checks / archived`, so the merge waits, shown as an expected check, and nothing turns red. A failure of the verdict itself is red. The hold applies only to a pull request into the default branch; a pull request into another pull request's branch reports `checks / archived` and may keep its change open.
- `github-checks.md` records the principle for every check: no check fails as a standing state, because a check red for days floods the maintainer's notifications and hides real failures; a valid pull request that may not merge into the default branch yet waits on a withheld required check.
- `checks / spec` on a pull request keeps strict validation and the `openspec-*` sync, and reports an unarchived change as a warning only; the `--draft` flag goes. On a push to `main` an unarchived change still fails.
- The knowledge files, `ARCHITECTURE.md`, `CONTRIBUTING.md`, and the `change-workflow` skill describe the one state label and the archive gate.

## Skills touched

repository change.

## Installed behavior

None: no skill changes. On GitHub, each pull request in this repository shows one specification state, and a pull request waiting for its freeze shows a pending required check instead of a red one; task progress is read locally with `just spec-changes status`.

## Impact

- Files: `.github/workflows/spec-labels.yml`, `.github/workflows/checks.yml`, `.github/labels.json`; `scripts/spec_approval.ts`, `spec_approval_test.ts`, `spec_changes.ts`, `spec_changes_test.ts`, `validate_harness.ts`, `validate_harness_test.ts`; `.agents/knowledge/spec-workflow.md`, `github-workflow.md`, `github-checks.md`, `github-settings.md`, `agent-authority.md`; `ARCHITECTURE.md`; `.github/CONTRIBUTING.md`; `.agents/skills/change-workflow/SKILL.md`; `justfile` where it describes the `spec-check` and `spec-changes` recipes.
- Untouched: everything under `skills/` (the `openspec-workflow` and `spec-kit-workflow` skills keep delivering their own label and check assets), `openspec/specs/`, archived changes.
- GitHub, after merge, by the maintainer: add `checks / archived` to the ruleset's required checks once the name is live on `main` (until then nothing but review stops an unarchived merge); prune the four retired labels (`sync_labels.ts --apply --prune`), which no workflow does. `labels / sync` creates `spec/pending` by itself.
- Sequencing: #122 also edits `spec_changes.ts`, `spec-workflow.md`, `github-checks.md`, `agent-authority.md`, and `ARCHITECTURE.md`; this branch rebases onto `main` once #122 merges.

## Non-goals

- No change to the public skills' label or check assets or their specs.
- No change to what the approval package is, to the record comment's format, or to strict validation.
- Removing the `snapshot` command, which loses its last workflow caller with this change and #122: a separate decision.

## Deliverable and pre-authorized actions

A pull request from `chore/single-spec-label`. Pre-authorized by the maintainer in conversation: pushes to that branch, the draft pull request and edits to its body.

## Tracked work

None; requested in conversation after #122's labels.
