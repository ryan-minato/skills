## Why

The `label-sync-workflow-asset` change gives `meta-github-workflow` a label-sync workflow asset shaped on this repository's own `.github/workflows/labels-sync.yml`. Once both exist, nothing says they are a pair: a fix to one step, such as the report of prune candidates or the branch guard, can land in one file while the other keeps the old behavior. The synchronization register is where this repository records pairs no script checks.

## What Changes

- **Synchronization register.** `.agents/knowledge/harness-maintenance.md` gains one row, beside the `openspec-workflow` asset row:
  - Source: `skills/meta/meta-github-workflow/assets/workflow-label-sync.yml`.
  - Mirror: `.github/workflows/labels-sync.yml`, with the placeholders resolved and the same triggers, job grants, branch guard, and apply and report steps, apart from two differences the row names: the schedule, optional in the asset and kept here, and the asset's GitHub Enterprise Server environment lines in the apply step (the host derived from the run's server URL and the enterprise token variable), which this repository's github.com workflow does not carry. The row says in its own words that nothing keeps the two files identical.
  - When it changes: the asset's triggers, grants, guard, or steps change, or this repository's workflow changes one of them.
  - Owner: author.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes. An agent that edits the label-sync asset, or this repository's `labels / sync` workflow, finds the other file named in the register and carries a step change across, or records why it does not.

## Impact

- `.agents/knowledge/harness-maintenance.md` only.
- `.github/workflows/labels-sync.yml`, `scripts/sync_labels.py`, `scripts/validate_harness.py`, and every other knowledge file stay as they are. No mechanical check is added.

## Non-goals

- A validator check keeping the two files identical. Management code is owned where it runs, and nothing binds a delivered asset to this repository's copy.
- Changing this repository's workflow to match the new asset, such as the checkout pin form or the GitHub Enterprise Server environment lines.

## Tracked work

Issue #90.
