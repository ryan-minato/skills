## Why

The `brownfield-suite` change adds five `engineering` skills that must depend on one another and share one evidence vocabulary. Three pieces of repository harness have to change for that:

- **Dependency grant.** The catalog rules forbid dependencies between `engineering` skills.
- **Routing.** Routing between the catalogs sends every brownfield request to `spec-driven-development`.
- **Duplicated material.** Nothing keeps five copies of the same material identical.

## What Changes

- **`skills/engineering/CONTEXT.md`:**
  - The class list names the brownfield suite.
  - `## Dependencies` grants the five `brownfield-*` members dependencies on one another by name. The grant overrides the "installed one at a time" rule for them. A missing member is installed by name through `ryan-minato-skills-installing`, and a handoff names every missing member. The section records the shared `## Evidence discipline` section and `references/records.md` as validator-enforced duplication, not a dependency.
  - `## Naming` records `brownfield-` as a suite prefix, not a catalog prefix.
  - `## Disambiguation` splits the current "converting a … brownfield codebase → `spec-driven-development`" route:
    - understanding, onboarding, contract recovery, and migration equivalence for an existing codebase go to the suite;
    - adopting the specification loop goes to `spec-driven-development`.

    It also resolves the now-ambiguous "the two skills here".
- **`skills/sdd/CONTEXT.md`:** `## Scope` and `## Disambiguation` point understanding existing code, recovering its contracts, and migration equivalence to the `engineering` brownfield suite.
- **`.agents/knowledge/skill-quality.md`:** the current-grants sentence names the `engineering` brownfield suite's grant. The synchronization register row for a `## Dependencies` grant requires this.
- **`ARCHITECTURE.md`:** the `engineering` bullet in `## Catalogs` names the brownfield suite and its internal grant, which is the same register row's second mirror.
- **`skills/engineering/README.md` and `README.zh.md`:** the introduction names the suite and shows one install line with all five members. The two files stay content-identical.
- **`scripts/validate_harness.py`:** a new check, listed in the module docstring, fails when a `skills/engineering/brownfield-*` member's `## Evidence discipline` section or its `references/records.md` differs from `brownfield-investigation`'s. The check also fails when a member lacks either item, or when members exist without the source member. It passes when no member exists yet.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes. Agents working in this repository:
- route brownfield requests to the suite when authoring or reviewing skills;
- accept dependencies between the five members as in range;
- get a failing `just validate`, naming the file to fix, when the shared material drifts.

## Impact

- The files listed above. Nothing else in `scripts/` changes.
- Workflows, labels, issue forms, the root README pair, `marketplace.json`, and `.agents/knowledge/harness-maintenance.md` do not change. The new check is a mechanically checked pair, which the register points to the validator docstring for.
- The installer skill and its script stay as they are: they already install several named skills in one run.

## Non-goals

- A catalog name prefix in `CATALOG_NAME_PREFIXES`. The existing `engineering` skills would fail it.
- A suite variant of the handoff template in `skill-quality.md`. Such a variant would also require checking the installer's description triggers against it.
- Any change to the skills named in the routing text.

## Tracked work

Issue #92.
