## Why

The skill change `ml-standard-alignment` on this branch makes `scaffold-ml`
name the durable `machine-learning` skills by role and creates two
mirrored pieces of content across catalogs: the run manifest module and
the research-spec section headings. The `scaffold` catalog's context must
say how a scaffold names skills of another catalog it may not depend on,
and the maintenance register must list every mirror so `just check`
cannot silently let them drift.

## What Changes

- `skills/scaffold/CONTEXT.md` `## Dependencies`: one sentence that
  durable topic skills of another catalog are outside the range, are
  named by role with a fallback the deposit carries, and are handed off
  through the installing skill.
- `.agents/knowledge/harness-maintenance.md` register rows: the run
  manifest module (`skills/machine-learning/experiment-provenance/assets/run_manifest.py`
  ↔ `skills/scaffold/scaffold-ml/assets/run_manifest.py`, byte-identical
  apart from the docstring's first line) and the research-spec section
  headings (`skills/machine-learning/research-workflow/assets/research-spec.md`
  ↔ the `research-task` schema template under
  `skills/scaffold/scaffold-ml/assets/`), each with its trigger and owner.
- `.claude-plugin/marketplace.json`: the `scaffold` plugin description,
  if the skill change moves its wording.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository find the cross-catalog naming rule in
the scaffold catalog's context and both mirrors in the register with
their triggers, so a change to one side is checked against the other.

## Impact

- Edited: `skills/scaffold/CONTEXT.md`,
  `.agents/knowledge/harness-maintenance.md`,
  `.claude-plugin/marketplace.json` (description field only, if at all).
- Unchanged: `scripts/validate_harness.py` (no mechanical mirror check is
  added; the register rows are the sync mechanism), the validators, the
  workflows, the rulesets, and every `meta` file.

## Non-goals

- A mechanical byte-identity check for the manifest module in
  `scripts/validate_harness.py`; a later change if the mirror drifts.
- Any change to the `machine-learning` or `meta` catalog's context.

## Tracked work

No issue: companion of `ml-standard-alignment`, planned in conversation.
