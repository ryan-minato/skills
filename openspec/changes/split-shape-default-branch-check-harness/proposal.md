## Why

The `split-shape-default-branch-check` change makes `openspec-workflow`'s check job pass the recorded request shape on its push branch too. `.agents/knowledge/harness-maintenance.md` registers that job asset against the `checks / spec` job of this repository's `checks.yml`, and the asset's command line against this repository's `scripts/spec_changes.py`. Both mirrors fall out of step unless they move with it. The push step would run `check --all` on the script's default, while the knowledge says the shape is pinned explicitly.

## What Changes

- **Push step pinned.** `.github/workflows/checks.yml`: the `checks / spec` push branch runs `just spec-changes check --all --shape combined`, and the step comment says both branches pin the combined shape. This repository is combined-shape, so the verdict does not change.
- **Help text.** `scripts/spec_changes.py`: the `check --all` help names both shapes, as the asset's does. No logic change.
- **Knowledge.**
  - `.agents/knowledge/github-checks.md`: the `checks / spec` row shows the pinned push command, and its healthy-run text says both commands pin `--shape combined`.
  - `.agents/knowledge/spec-workflow.md`: `## Archive executor and the freeze` names the push check beside the `spec-check` recipe as carrying the explicit pin; `## Verification in checks` shows the pinned push command.
  - `.agents/knowledge/harness-maintenance.md`: the `scripts/spec_changes.py` row names the push check's `--shape combined` flag in `checks.yml` beside the recipe's.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes. Agents working in this repository read a push check that states its shape, and knowledge that matches it. The check's verdict on `main` is unchanged: any change outside `openspec/changes/archive/` fails.

## Impact

- `.github/workflows/checks.yml`, `scripts/spec_changes.py`.
- `.agents/knowledge/github-checks.md`, `spec-workflow.md`, `harness-maintenance.md`.
- No change to the `justfile`: `spec-changes` passes its arguments through. No change to `ARCHITECTURE.md`, which does not quote the push command. No required-check name changes.

## Non-goals

- A new `just` recipe for the push check.
- This repository's request shape, which stays combined.
- `spec-workflow.md`'s `## Request automation` sentence that a push to `main` fails on any change outside `archive/`. It stays true for a combined-shape repository.

## Tracked work

Issue #99.
