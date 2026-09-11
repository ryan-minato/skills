## Why

The skill change `ml-standard-alignment` on this branch makes `scaffold-ml`
name the durable `machine-learning` skills by role and creates three
mirrored pieces of content across catalogs: the run manifest module, the
research-spec section headings, and the canonical run-record field list.
The `scaffold` catalog's context must say how a scaffold names skills of
another catalog it may not depend on, and the maintenance register must
list every mirror so `just check` cannot silently let them drift.

## What Changes

- `skills/scaffold/CONTEXT.md` `## Dependencies`: one sentence that
  durable topic skills of another catalog are outside the range, are
  named by role with a fallback the deposit carries, and are handed off
  through the installing skill.
- `.agents/knowledge/harness-maintenance.md` register rows: the run
  manifest module (`skills/machine-learning/experiment-provenance/assets/run_manifest.py`
  ↔ `skills/scaffold/scaffold-ml/assets/run_manifest.py`, byte-identical
  apart from the docstring's first line); the research-spec section
  headings (`skills/machine-learning/research-workflow/assets/research-spec.md`
  ↔ `skills/meta/meta-spec-workflow/assets/openspec/research-task/templates/research.md`
  ↔ the fallback skeleton in `skills/scaffold/scaffold-ml/assets/agents-md.md`);
  the canonical run-record field list
  (`skills/machine-learning/experiment-provenance/references/run-record.md`
  ↔ `skills/meta/meta-github-workflow/assets/experiment-record.md` ↔
  `skills/meta/meta-gitlab-workflow/references/mlops.md`), each with its
  trigger and owner.
- `.claude-plugin/marketplace.json`: the `scaffold` and `meta` plugin
  descriptions, if the skill change moves their wording.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository find the cross-catalog naming rule in
the scaffold catalog's context and every new mirror in the register with
its trigger, so a change to one side is checked against the other.

## Impact

- Edited: `skills/scaffold/CONTEXT.md`,
  `.agents/knowledge/harness-maintenance.md`,
  `.claude-plugin/marketplace.json` (description fields only, if at all).
- Unchanged: `scripts/validate_harness.py` (no mechanical mirror check is
  added; the register rows are the sync mechanism), the validators, the
  workflows, the rulesets.

## Non-goals

- A mechanical byte-identity check for the manifest module in
  `scripts/validate_harness.py`; a later change if the mirror drifts.
- Any change to the `machine-learning` catalog's context.

## Tracked work

No issue: companion of `ml-standard-alignment`, planned in conversation.
