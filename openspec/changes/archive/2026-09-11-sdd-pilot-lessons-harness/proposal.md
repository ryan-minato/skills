## Why

The skill change `sdd-pilot-lessons` on this branch moves the archive
script into `meta-spec-workflow`, introduces paradigm builders beside
contract and platform builders, and makes the discussion-closed approval
gate the default the spec-driven skills teach. This repository's own
harness is generated from those builders and pins the script's origin, so
it must follow in the same pull request: the mirror's origin path, the
register rows, the contract-flow rule the reviewer applies, the catalog
description in `ARCHITECTURE.md`, and — by the maintainer's decision — its
own specification gate, which still requires the fixed comment.

## What Changes

- Archive mirror: `scripts/validate_harness.py` compares
  `scripts/archive_completed_changes.py` with the copy in
  `skills/meta/meta-spec-workflow/scripts/`; its docstring names both
  origins correctly; the row in `harness-maintenance.md` and the header
  comment of `.github/workflows/spec-archive.yml` say the same.
- Specification gate: this repository's contract switches to
  discussion-closed — the draft opens at the record, the agent stops, the
  maintainer discusses on the pull request and directs record edits in
  conversation, closes the discussion in conversation, and the agent
  reconciles review threads and the record before design and
  implementation; the blocking comment is no longer required.
  `spec-workflow.md` Lifecycle and tracked-work bullet, `change-workflow`
  §3 and §7, the pull request template's approval line and checklist
  item, and `agent-authority.md`'s ready condition say so.
  `scripts/check_pr_policy.py` is unchanged: its `Phase:` rule holds in
  both modes.
- Purpose-line exception: `spec-workflow.md` records that a change which
  removes or reshapes a domain's capabilities corrects that domain's
  `## Purpose` line by hand in the same pull request, named in its
  proposal.
- Contract flow: the `code-review` project skill and `ARCHITECTURE.md`
  name paradigm builders as a third kind; the register gains the row
  pairing the platform builders' extension-slot lists with
  `meta-spec-workflow`'s slot tables; the `meta` plugin description in
  `marketplace.json` follows.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository stop after publishing a change record,
wait for the maintainer to close the discussion, reconcile the pull
request's threads before implementing, and find the archive script's
origin and the paradigm-builder rule where the register and reviewer
expect them.

## Impact

- Edited: `scripts/validate_harness.py`,
  `.agents/knowledge/harness-maintenance.md`,
  `.github/workflows/spec-archive.yml` (comment only),
  `.agents/knowledge/spec-workflow.md`,
  `.agents/skills/change-workflow/SKILL.md`,
  `.github/PULL_REQUEST_TEMPLATE.md`,
  `.agents/knowledge/agent-authority.md`,
  `.agents/skills/code-review/SKILL.md`, `ARCHITECTURE.md`,
  `.claude-plugin/marketplace.json` (description field only).
- Hand-edited main specs: the `## Purpose` lines of
  `openspec/specs/meta/meta-github-workflow/spec.md` and
  `openspec/specs/meta/meta-gitlab-workflow/spec.md`.

## Non-goals

- The push path for `spec-archive.yml` and the three sentences that still
  name the Actions identity (`automated-archive`).
- Any change to `scripts/check_pr_policy.py` or the required checks.

## Tracked work

No issue: companion of `sdd-pilot-lessons`, planned in conversation.
