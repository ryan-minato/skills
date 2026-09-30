## Why

This repository runs its own copy of the `openspec-workflow` comment-command workflow, and the register in `.agents/knowledge/harness-maintenance.md` keeps it on the asset's steps. It has the same defects the `spec-command-reply` change fixes in the asset (#96, #100): an admitted `/specs` or CRLF `/spec` comment turns the run red with no reply, `/spec status` ending in a carriage return gets the help text, and the reply can name and link a different head than its content came from.

## What Changes

- **`.github/workflows/spec-command.yml`** takes the asset's new steps, with its own pin and header kept:
  - the job runs only for a comment that begins with `/spec` followed by the end of the comment, a space, a tab, or a line break;
  - every admitted command gets a reply: a trailing carriage return is ignored, a tab separates words, anything else is answered with the command list, and the header echoes the command as it was read;
  - the reply names and links the head recorded in the snapshot, and the pull request's state still comes from `gh pr view`;
  - the `pull-requests: read` comment no longer says the job reads "the request and its files".
- **`.agents/knowledge/github-checks.md`**, the `spec / command` row: "a pull request comment starting with `/spec`" becomes the admitted shapes above, and "skipped for other comments" stays.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes here; the skill change is `spec-command-reply`. Agents and collaborators working in this repository get a reply to every `/spec` command the workflow runs for, none for `/specs` or `/speckit.*` words, and a reply whose head and links match its content. An agent diagnosing `spec / command` reads the admission rule from `github-checks.md`.

## Impact

- `.github/workflows/spec-command.yml`, `.agents/knowledge/github-checks.md`.
- `scripts/validate_harness.py` checks only job names against `github-checks.md`; the job name `spec / command` does not change.
- `.agents/knowledge/harness-maintenance.md`, `spec-workflow.md` `## Request automation`, and `ARCHITECTURE.md` stay accurate as written.
- The workflow runs from `main` only, so its live behavior changes after the merge.

## Non-goals

- `scripts/spec_changes.py` and the snapshot schema.
- The other workflows, including `spec-labels.yml`, which neither parses comments nor prints the head.

## Tracked work

Issues #96 and #100.
