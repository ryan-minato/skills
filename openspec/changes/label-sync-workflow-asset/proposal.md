## Why

A repository built with `meta-github-workflow` commits its label taxonomy as `labels.json` and gets its own `scripts/sync_labels.py`, but nothing runs that script after the build. Every later taxonomy change needs someone to dry-run, review, and apply it by hand, and until they do the forms, the labeler, and the release notes reference labels that do not exist. This repository already solved that for itself with the `labels / sync` workflow; the builder should offer the same.

## What Changes

- `meta-github-workflow` gains a label-sync workflow asset, delivered as the target's `.github/workflows/labels-sync.yml` with the job `labels / sync`.
  - It runs the delivered `scripts/sync_labels.py --apply` on a push to the default branch that touches the labels file, the sync script, or the workflow, and on manual dispatch.
  - It runs only on the default branch, and its job holds only `issues: write` and `contents: read`.
  - It runs unchanged on GitHub Enterprise Server: the apply step derives the host and the token variable gh reads for it from the run's server URL.
  - It never deletes a label. A label on the repository but absent from `labels.json` is reported as a warning in a run that stays green; pruning stays a manual, authorized step.
  - It is recorded in the target's checks knowledge with its healthy-run shape and is never a required check.
- The builder offers the workflow as a numbered frontier question whenever the plan commits a `labels.json`, recommending yes, including on a solo repository.
  - A weekly schedule is included only when the user confirms `labels.json` is the sole source of labels and names the schedule's owner.
  - A decline delivers no workflow, and the knowledge records the manual sync path.
- The build-time dry run, review, authorized apply, and readback stay the first application of the approved taxonomy.
  - The workflow reaches the default branch only once that apply was authorized, because its first push-triggered run would otherwise apply the taxonomy unreviewed.
  - Before it lands, the builder tells the user that each later merged change to `labels.json` is applied without a further prompt and that nothing is deleted. The handoff presents the first run as a readback expected to be all-skip.
- The delivered-workflows scenario of the management-code requirement covers the new workflow: its report step is a marked deliberate deferral, and a failing sync fails the run.

## Skills touched

- `meta/meta-github-workflow` (modified): the label-sync workflow offered, delivered, and sequenced after the authorized build-time apply; the delivered-workflows scenario extended to it.

## Installed behavior

An agent running the builder now offers a workflow that keeps the repository's labels in step with `labels.json` after the build, and delivers it when the user accepts. Before, it delivered only the script and left every later sync to a person. This adds a capability → `feat`.

## Impact

- No description, `compatibility` field, symlink, `marketplace.json` entry, or catalog README row changes. The `meta` README row already says "labels extending the defaults", which stays true, so the `README.md` / `README.zh.md` pair is untouched.
- `skills/meta/meta-github-workflow/references/durable-harness.md` changes outside its `## Extension slots` table, so the slot mirror in `meta-spec-workflow` (register row in `.agents/knowledge/harness-maintenance.md`) does not move.
- The companion repository change `label-sync-workflow-asset-harness` registers the pair this change creates: the new asset and this repository's own `.github/workflows/labels-sync.yml`, which served as its reference shape.
- This repository's `scripts/sync_labels.py`, `.github/workflows/labels-sync.yml`, and the skill's `assets/sync_labels.py` do not change.

## Non-goals

- Deleting labels automatically, in the workflow or on a schedule.
- Organization-level taxonomy (`sync_org_taxonomy.py`, issue types and fields).
- A dry-run preview on pull requests: the sync script rejects a malformed file before any write, and a merged change is applied by design. A solo repository therefore has no pull-request-side validation or preview of `labels.json`; the taxonomy check covers it only where the `actions-automation.md` branch is selected.
- A GitLab parallel in `meta-gitlab-workflow`, which ships no target-side label-sync script yet; a follow-up issue.
- The checkout pin form of the existing workflow assets.

## Tracked work

Issue #90.
