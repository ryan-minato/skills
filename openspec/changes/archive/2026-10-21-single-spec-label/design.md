## Context

`spec-labels.yml` (`pull_request_target`, main's copy, `pull-requests: write`) runs three steps: `scripts/spec_approval.ts --apply` decides `spec/approved` from the label history, the record comment, and the approval package at the head; `spec_changes.ts snapshot` reads the head through the API; `spec_changes.ts labels` plans the archive axis and the progress axis from the related changes, and a shell loop applies the plan against a literal `case` list. `validate_harness.ts` (`spec`) checks `labels.json` against `MANAGED_LABELS` from `spec_changes.ts` and that `case` list.

`checks.yml` (`pull_request`, the pull request's own copy, read-only token) runs `checks / spec`: strict validation, then `spec_changes.ts check`, which fails a ready pull request holding an unarchived related change (a warning on a draft, `--draft`), then the `openspec-*` sync; `checks / gate` aggregates it and is the required check. `pr-policy.yml` already runs the base's copy of its script, checking out the base and fetching `refs/pull/<n>/head`.

The reference is hoshiori-dev/devcontainer-features: `scripts/sync_pr_labels.ts` keeps one state label of `pending | approved | archived`, or none; `pr.yml` splits an `archive-verdict` job from a carrier job named `spec-archived` when no change is unarchived and `awaiting-archive` otherwise, with `spec-archived` required. Binding constraint: nothing under `skills/` changes. See proposal.md for motivation.

## The state label

Decided by `spec_approval.ts` from the pull request's present state, in this order:

| The pull request holds an unarchived change of its own | Approval holds (rules unchanged) | Pull request adds a path under `openspec/changes/archive/` | State |
|---|---|---|---|
| yes | yes | — | `spec/approved` |
| yes | no | — | `spec/pending` |
| no | — | yes | `spec/archived` |
| no | — | no | none |
| no | — | file list cut short | `spec/archived` left as it is; the other two removed |

- *Unarchived change of its own*: a directory directly under `openspec/changes/` other than `archive/` at the head whose entries the base commit (`pull.base.sha`, one more tree read) does not hold identically: new, or edited by this pull request. A change a stacked base already carries belongs to the pull request that brought it, so a pull request into another pull request's branch neither inherits its parent's `spec/pending` nor has its approval bound to the parent's package. The package comparison takes the own changes at the head, plus those of the recorded commit that the present base does not hold unarchived.
- *Adds a path*: a file of the pull request's file list (`pulls/<n>/files`) with status `added` or `renamed` under the archive directory. The list is cut short when it holds fewer entries than the pull request's `changed_files`, or that count exceeds 3000, the API's limit.
- *Approval holds* binds the approval to the package at the recorded commit. A change to the package after that commit (any file of an unarchived change except `tasks.md`, or the main spec under a delta; an archive commit is not one) ends it: the next run sets `spec/pending` in place of `spec/approved` and marks the record withdrawn, and the maintainer approves the new package by adding `spec/approved` again. The state never stays at `spec/approved` on a fact the run cannot confirm.
- One more withdrawal: a person adding `spec/pending` after the latest adding of `spec/approved` withdraws the approval, as a person removing `spec/approved` does. The withdrawal holds after the archive too: when the pull request holds no change of its own but archives one, a person's withdrawal after the latest adding of `spec/approved` gives `spec/pending` and marks every live record withdrawn, until the maintainer adds `spec/approved` again.
- Writes: the record and withdrawn records first, as now; then the wanted state label is added, then every other state label on the pull request is removed. Labels outside the three are never touched.
- After an error, on an open pull request: add `spec/pending`, remove `spec/archived` if the pull request carried it, then remove `spec/approved`; on a closed one: remove `spec/approved`. One state label remains either way. The workflow's failure step keeps removing `spec/approved`.
- Read-only `just spec-approval <n>` prints the state, the reason, and the writes; exit 0 for `spec/approved`, 1 for any other state.

## The archive gate

The principle, recorded in `github-checks.md` as a rule for every check: no check fails as a standing state. A pull request that is valid but may not merge yet, here because its change waits for the freeze, is held by a withheld required check that shows as pending; a failure means something to fix now. A check that stays red for days fills the maintainer's notifications and hides real failures behind the same mark.

The hold applies only to a pull request into the default branch, the one long-lived branch. A pull request into another pull request's branch may keep its change unarchived: the change is archived on the request that reaches `main`.

Two jobs in `checks.yml`, on `pull_request` only, outside `checks / gate`:

- `checks / archive-verdict`: when the pull request's base branch is not the default branch (`github.event.pull_request.base.ref` against `github.event.repository.default_branch`, both through `env:`), writes `archived=true` and says that the archive gate holds only requests into the default branch. Otherwise it checks out the base commit (`ref: base.sha`, `fetch-depth: 0`, no credentials), fetches `refs/pull/<n>/head`, installs Deno, and runs the base's `./scripts/spec_changes.ts status --base <base> --head <head> --json` into a file under `$RUNNER_TEMP`. A non-zero exit fails the job. `jq` then writes `archived=true` when no related change has `"state": "active"` and `archived=false` otherwise; nothing else from the file reaches `$GITHUB_OUTPUT`, so a change name the pull request chose never does. The step prints the related changes' table from `status` without `--json` for the reader.
- The carrier job: `needs: archive-verdict`, named `${{ needs.archive-verdict.outputs.archived == 'true' && 'checks / archived' || 'checks / awaiting-archive' }}`, one step that says which, and succeeds either way. A failed verdict skips it, so neither name succeeds.

The ruleset requires `checks / archived`. While a change is unarchived, GitHub shows the required check as expected and the merge waits; the run is green. The verdict depends on the commits and the base branch alone, never on the draft state.

`checks / spec` on a pull request: strict validation and the sync as now; `spec_changes.ts check` with a request source reports every unarchived related change as a warning naming the archive gate and never as a finding, and `--draft` is removed. With `--all` (a push to `main`) an unarchived change stays a finding.

## Placement

| Item | File and section | Check that proves it |
|---|---|---|
| State decision and writes | `scripts/spec_approval.ts`: `STATE_LABELS`, `decide`, `writesOf`, `failClosed`, the `Api` label calls taking one of `STATE_LABELS`, a `files` read; header and usage | `just test` (`spec_approval_test.ts`: one case per table row, the `spec/pending` withdrawal, write order, fail-closed on open and closed) |
| Labels workflow | `.github/workflows/spec-labels.yml`: the snapshot and plan steps removed; header | `spec_approval_test.ts` assertions on the file; `just validate` (`workflows`) |
| Taxonomy | `.github/labels.json`: three `spec/*` rows; `validate_harness.ts` `spec` check imports `STATE_LABELS`, drops the `case`-list check | `just validate`; `validate_harness_test.ts` |
| `labels` command and `--draft` removed | `scripts/spec_changes.ts` command table, parser, taxonomy constants, `cmdCheck`, usage; `spec_changes_test.ts`; `justfile` comments | `just test`; `spec_changes.ts labels` and `check --draft` exit 2 |
| Archive gate | `.github/workflows/checks.yml`: the two jobs, the `checks / spec` step without `--draft`; header | `just validate` (`checks`, `workflows`); `validate_harness.ts` `checks` check reads a name of the form `${{ <cond> && '<a>' \|\| '<b>' }}` as the two names `<a>` and `<b>`; `validate_harness_test.ts` |
| Required checks | `github-checks.md` table, healthy runs, `## Diagnosing a red check` item 3; `github-settings.md` required-checks row | `just validate` (`checks`) |
| The no-standing-failure rule | `github-checks.md`: a rule beside the required checks — a waiting state is a withheld required check, held only for the default branch, never a failure | review; the `git grep` below finds no document that still calls a red check the expected state |
| Docs | `spec-workflow.md` (lifecycle, `## Approval gates`, `## Archive executor and the freeze`, `## Request automation`); `github-workflow.md` label table; `agent-authority.md` ready conditions; `ARCHITECTURE.md`; `CONTRIBUTING.md`; `change-workflow` §5 | `git grep -nE 'spec/(unarchived\|not-started\|in-progress\|done)\|red until the freeze\|red for the whole'` outside `skills/`, `openspec/specs/`, archives returns nothing |

## External impact

- `skills/**`, `openspec/specs/**`, archived changes: unchanged; `git diff --stat origin/main -- skills openspec/specs openspec/changes/archive` is empty.
- GitHub, after merge: the maintainer adds `checks / archived` to ruleset `19602018`'s required checks and reads it back (`github-settings.md ## Applying and reading back`); `labels / sync` creates `spec/pending`; the maintainer prunes the four retired labels. The first `spec / labels` run on each open pull request replaces its state labels; retired labels stay until the prune.

## Decisions

- **One script, `spec_approval.ts`, for the state label**: it already reads the head tree and owns the fail-closed path; the state needs one more read (the file list). Rejected: keeping `spec_changes.ts labels` for `spec/archived` (two writers of one label set reintroduce the problem); renaming the script (churn in the recipe, the deny-safe read path, and every document, for no behavior).
- **The pull request's own changes, against its base**: a pull request into `main` holds only its own unarchived changes (`checks / spec` keeps `main` free of them), but a stacked base carries its parent's; reading every change of the head tree would give a child the parent's state and couple their approvals. Adopted on review of the Codex finding on #123 (plan A, approved by the maintainer in conversation), with the fail-closed and post-archive withdrawal fixes below. Rejected: the head tree alone (wrong for stacked bases, which this change permits); the diff's related changes (a file list cut short would leave a change unseen, and the base tree is one read).
- **Drop the progress axis**: `tasks.md` and `just spec-changes status` show it; no rule reads the labels.
- **`spec/archived` replaces `spec/approved` after the freeze**: nothing after the freeze reads the label; the record comment stays as the approval's evidence.
- **A withheld required check, not a failing one**: waiting for the freeze is a state, so it shows as pending and keeps `checks / gate` meaningful. Rejected: naming `checks / gate` conditionally (a real failure and a wait would look alike); a commit status posted by `spec / labels` (a write token for what a read-only run decides).
- **The default branch only**: GitHub Flow has one long-lived branch, and a stacked pull request merges into a branch that is itself reviewed before `main`; holding it would force an archive before the work is whole. Rejected: holding every base (blocks stacking); relying on the ruleset targeting `main` alone (the check would still read `awaiting-archive` on a stacked request, which misstates it).
- **The verdict from the base's `status --json`**: `status` exists on `main` today, so the gate works on this pull request without a bootstrap fallback, and the base's copy keeps a pull request from relaxing its own verdict by editing the script. The workflow file itself still comes from the pull request; review guards it, as in the reference.
- **Outside the gate**: the verdict runs on pull requests only; `ci_gate.ts` fails a skipped dependency, so on a push to `main` the gate would go red. A failed verdict is red on its own job and withholds `checks / archived`.
- **No migration code**: the prune is one maintainer command; a script that deletes retired labels would widen what the token writes.

## Risks / Trade-offs

- [Between the merge and the ruleset update, an unarchived pull request can merge] → the maintainer updates the ruleset right after the merge; the order (name live on `main`, then the ruleset) is the one `github-checks.md` prescribes.
- [Retargeting a pull request's base fires only `edited`, which starts no `checks.yml` run, so a pull request retargeted onto `main` keeps the verdict of its last push — `checks / archived` from its stacked days — with its change unarchived] → accepted by the maintainer during implementation: the gate stays in `checks.yml`, `github-checks.md ## No standing failure` says to rerun `checks` after retargeting onto `main`, and #127 tracks the fix. Rejected: `edited` on `checks.yml` (every body edit reruns the quality gate, and skipped required jobs report success).
- [A pull request edits `checks.yml` to report `checks / archived` regardless] → review of workflow edits, as for every `pull_request` workflow; the verdict's script is the base's.
- [An archive commit is reverted after the freeze] → `spec/approved` is no longer on the pull request, so the state is `spec/pending` and the record is withdrawn; the maintainer approves again. Fail closed, rare.
- [The pull request lists more than 3000 files] → `spec/archived` is left as it is; the reason says so. The archive gate reads git, not the API, and is unaffected.
- [Merge order with #122] → this branch rebases after #122 merges and resolves the overlapping edits.

## Verification plan

Script and tool harnesses:
- `just check` passes; `just test` covers every row of the state table, a package change after the recorded commit (`spec/approved` → `spec/pending`, record withdrawn), a `tasks.md`-only change and an archive commit (approval kept), a stacked base whose change is not the pull request's own (no inherited state, approval unbound from the parent), a withdrawal after the archive, failing closed on an archived pull request, the `spec/pending` withdrawal, the write order, both fail-closed paths, `check` warning instead of failing on an unarchived related change, and the validator's reading of a conditional job name.
- Read-only `./scripts/spec_approval.ts <n>` from this branch against #122 and against this pull request prints one state each and the writes it would make, with no write.
- On this pull request, which holds an unarchived change: `checks / archive-verdict` succeeds with `archived=false`, `checks / awaiting-archive` succeeds, no `checks / archived` is reported, `checks / spec` and `checks / gate` are green. After the archive commit: `checks / archived` succeeds.
- A throwaway draft pull request into `chore/single-spec-label` holding an unarchived change reports `checks / archived` without running `status`; it is closed and its branch deleted afterwards.
- `./scripts/spec_changes.ts labels` and `check --draft` exit 2; `just spec-changes status` and `just spec-check origin/main HEAD` still work.
- The `git grep` in Placement returns nothing; `git diff --stat origin/main -- skills openspec/specs openspec/changes/archive` is empty.

Skipped:
- A live `spec / labels` run with the new code before merge: the workflow runs `main`'s copy, so the first live run is on the next pull request event after merge.
- The merge block itself before the ruleset update: the ruleset changes only after merge; until then the pending check is shown but not required.
