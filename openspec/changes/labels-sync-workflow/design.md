## Context

See proposal.md. `scripts/sync_labels.py` already does the work. It reads
`.github/labels.json`, lists the repository's labels through `gh`, prints a
JSON plan (`create`, `update`, `skip`, `prune_candidates`, `pruned`), and
with `--apply` creates and edits labels, deleting only with `--prune`. It
is idempotent. It must stay byte-identical to
`skills/meta/meta-github-workflow/scripts/sync_labels.py` (the copies check
in `scripts/validate_harness.py`), so the workflow calls it unchanged.

The other workflows set the conventions the new one follows: a header
comment, `permissions: {}` at the top with job-level grants,
`actions/checkout` pinned by SHA, and no third-party action beyond
TruffleHog (`github-settings.md`, Actions row). `validate_harness.py`
requires every workflow job name to appear in the `github-checks.md` table
and the reverse. `github-settings.md` records that no workflow holds
`contents: write` and that no identity needs a ruleset bypass. Changing a
label through the REST API needs `issues: write`, which keeps both
statements true.

## Placement

| What Changes bullet | File | Check that proves it |
|---|---|---|
| The workflow | `.github/workflows/labels-sync.yml`, job `sync` named `labels / sync` | `just check` (job-name table check); a read-through against the conventions above; the summary step run locally against a recorded plan |
| Checks table row | `.agents/knowledge/github-checks.md` | `just validate` (`check_checks_doc`, both directions) |
| Settings register | `.agents/knowledge/github-settings.md` Labels row | read-through |
| Maintenance register | `.agents/knowledge/harness-maintenance.md` "a catalog added or removed" row | read-through; `just validate` |
| Architecture map | `ARCHITECTURE.md` `## GitHub Workflow` | read-through; `just validate` (pointers) |

## External impact

- The remote labels change when the workflow first runs on `main` after the
  merge. Before that run, a dry run
  (`python3 scripts/sync_labels.py --file .github/labels.json --repo ryan-minato/skills`)
  shows what it will do. If the plan is not all-skip, the first run applies
  it, and that is the intended outcome.
- The ruleset and its required checks are not edited. `labels / sync` runs
  only on `main` and never on a pull request, so it could not gate one.
- `scripts/`, `.github/labels.json`, and the skill's mirror are not edited.
  Proof: `git diff --stat origin/main...HEAD -- scripts .github/labels.json skills`
  is empty.

## Decisions

- **Report prune candidates, never delete them** (serves the workflow). A
  deletion cannot be undone and strips the label from every issue and pull
  request. `github-workflow.md` requires listing those issues and getting
  authorization first. The workflow emits one `::warning::` per candidate
  and a summary table, and it does not fail. Rejected: `--prune` in the
  workflow. Also rejected: failing the job on a candidate, because a red
  scheduled run would repeat every week until someone decided a deletion.
- **Triggers: path-filtered push to `main`, weekly schedule, manual
  dispatch** (serves the workflow). The push applies a taxonomy change as
  it merges. The schedule reverts edits made in the web interface, which
  the settings register already forbids ("exactly `.github/labels.json`").
  Dispatch covers the first run and repairs. GitHub disables schedules on a
  repository with 60 days of inactivity, but the push trigger still fires
  on every relevant change. Rejected: a dry run on pull requests (see the
  proposal's non-goals).
- **Run only on `refs/heads/main`** (serves the workflow). A job-level `if`
  prevents a dispatch from another branch from applying an unmerged
  taxonomy. The job runs only code already on `main` and handles no
  content from a pull request, so it has none of the `pull_request_target`
  exposure that `spec / labels` guards against.
- **Grant `issues: write` and `contents: read` only** (serves the
  workflow). Labels are an issues resource. Keeping `contents` read-only
  preserves the settings register's claim that no workflow needs a bypass.
- **No concurrency cancellation** (serves the workflow). With
  `cancel-in-progress: false`, a later run queues behind an earlier one
  instead of cancelling a half-applied plan. Because the script is
  idempotent, the queued run finishes the job.

## Risks / Trade-offs

- [A malformed `labels.json` reaches `main`] → `validate_harness.py` fails
  the pull request first. If it slips through, the script exits 2 before
  any write and the run turns red.
- [A rename in `labels.json` creates the new label and leaves the old one
  on its issues] → the old name shows up as a prune candidate warning, and
  the maintainer relabels and deletes it. The same holds today.
- [Anyone who can edit `labels.json` on `main` controls the labels] → only
  merges reach `main`, and the ruleset already governs merges.

## Verification plan

- The workflow: a read-through against the Context conventions (header
  comment, `permissions: {}`, job-level `contents: read` + `issues: write`,
  pinned checkout, the `main` guard, no `--prune`). The summary step's
  shell runs locally against the JSON of a real dry run and against a
  hand-made plan with two `prune_candidates` and one `update`. It must
  print one warning per candidate, write the Markdown summary to a scratch
  file, and exit 0. Local runs must not call `--apply`.
- The checks table: `just validate` passes with the row. Removing the row
  in a scratch copy makes it fail.
- The two registers and the architecture map: read-through. `just validate`
  passes.
- `git diff --stat origin/main...HEAD -- scripts .github/labels.json skills`
  is empty.
- `just check` at the end.
- After the merge (maintainer): dispatch the workflow once. The run must be
  green, and `gh label list` read back against `labels.json` must match.
  The date goes into the settings register's Last verification.

## Open Questions

None.
