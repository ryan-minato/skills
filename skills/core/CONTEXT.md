# core — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

## Requirements

- Skills here are installed globally and load into **every** session, so
  their `description` must be tightly scoped: a core skill that triggers on
  unrelated tasks pollutes all projects at once.
- Keep context cost minimal — SKILL.md bodies in this catalog should be
  short, with anything long pushed to `references/`.
- A skill belongs in `core` only if it is useful regardless of project type.
  If it only helps certain kinds of projects, it belongs in another catalog.

## Dependencies

- Skills here may depend only on other `core` skills. No grant to any other
  catalog: a core skill is installed globally, so anything it named from a
  per-project catalog would be absent in most sessions.
- Handoffs to skills outside `core` are optional, carry a fallback, and go
  through `ryan-minato-skills-installing`, this library's single install
  entry point; it ships here so every skill in the library can rely on it.
- No dependency on or recommendation of skills from other repositories; no
  exemptions.

## Naming

Default shape, no prefix or suffix: `<subject>-<action>` (`git-commit`,
`sensitivity-check`, `plan-clarification`). `ryan-minato-skills-installing`
carries an ownership prefix because it is bound to this library;
`meta-harness` is a proper noun for the meta-level harness methodology, not
a use of the `meta` catalog's prefix.

## Disambiguation

Clarifying, stress-testing, or pinning down a plan, an idea, or a decision
that is already on the table → `plan-clarification` · defining what something
should achieve from scratch and recording it as a goal document →
`goal-alignment` in `engineering` · resolving ambiguous requirements while
already writing code → `programming-guidelines` · producing the
implementation plan or task breakdown itself → none of them (out of catalog
scope).

## References

_(none yet — add catalog-scoped reference URLs here)_
