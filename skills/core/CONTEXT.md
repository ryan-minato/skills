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
