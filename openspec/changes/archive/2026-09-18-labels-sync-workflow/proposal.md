## Why

`.github/labels.json` is the label taxonomy, but GitHub only learns about a
change to it when the maintainer runs `scripts/sync_labels.py --apply` by
hand after the merge. Every catalog change so far has carried that step as a
maintainer action, and a label edited in the web interface drifts until
someone notices. A workflow can apply the file itself, so the step and the
drift both go away.

## What Changes

- New workflow `.github/workflows/labels-sync.yml`, job `labels / sync`:
  on a push to `main` that touches `.github/labels.json`,
  `scripts/sync_labels.py`, or the workflow itself, on a weekly schedule,
  and on manual dispatch, it runs `scripts/sync_labels.py --apply` against
  the repository. That creates missing labels and corrects the color and
  description of drifted ones. It never deletes a label: labels on GitHub
  that are absent from the file are listed as warnings in the run summary,
  and the authorized `--prune` stays manual.
- `.agents/knowledge/github-checks.md`: a `labels / sync` row.
- `.agents/knowledge/github-settings.md`: the Labels row names the workflow
  as the mechanism and keeps the manual command as the fallback.
- `.agents/knowledge/harness-maintenance.md`: the catalog row no longer
  lists a label sync after merge as a maintainer action.
- `ARCHITECTURE.md` `## GitHub Workflow`: one sentence on the workflow.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository stop handing the maintainer a label sync
after a `labels.json` change. The merge applies it. The only label work
left for the maintainer is deleting a label, and the run summary lists the
candidates for that.

## Impact

- Added: `.github/workflows/labels-sync.yml`.
- Edited: `.agents/knowledge/github-checks.md`,
  `.agents/knowledge/github-settings.md`,
  `.agents/knowledge/harness-maintenance.md`, `ARCHITECTURE.md`.
- Unchanged: `scripts/sync_labels.py` (a byte-identical mirror of the
  `meta-github-workflow` skill's script), `.github/labels.json`, the
  validators, the ruleset. The new job is not a required check.

## Non-goals

- Deleting labels automatically. A deletion strips the label from every
  issue and pull request, and `github-workflow.md` requires listing those
  and an authorization first.
- A dry-run preview on pull requests: `validate_harness.py` already checks
  the file's shape and its agreement with the forms.
- A label-sync workflow asset in the `meta-github-workflow` skill, which a
  separate issue tracks.

## Tracked work

No issue: planned in conversation.
