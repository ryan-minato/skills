# Durable Output

Read on every build, before depositing the contract.

## Where the contract lives

Deposit the adapted contract at `.agents/knowledge/project-workflow.md` in
the target project (create the directory if the project has no knowledge
tree; if the project keeps agent knowledge elsewhere, follow the existing
convention and record the path in the hand-off). This file is the single
source of truth for the management model: the governance builder reads it to
anchor authority boundaries, and the platform lifecycle builders read it to
map — never re-decide — its semantics.

## The entrypoint pointer

Add one pointer to the project's agent entrypoint (`AGENTS.md` or its
equivalent), event-triggered rather than always-read:

> Read `.agents/knowledge/project-workflow.md` before creating, splitting,
> or closing tracked work, before starting a change request, and before
> proposing any new management structure.

Do not paste contract content into the entrypoint; one source of truth per
fact.

## What the contract must carry

The deposited file, adapted from the asset, must state:

- Base profile, overlays, and change propagation, each with the selecting
  fact.
- Every enabled semantic with its deletion-test answer.
- Every intentionally omitted semantic with the concrete trigger that would
  justify enabling it — this is what makes progressive formalization real
  rather than a slogan.
- The work-decomposition rule, the objective/timebox policy, and the
  planning policy, phrased so an agent can apply them to a concrete item.
- An "Update this file when" list naming the events that reopen the design.

The contract records facts and policies. Planning surfaces are views over
it and may be rebuilt at any time; nothing in the contract may exist only
inside a view.

## Survival rules

- Platform-neutral forever: the file must read correctly with no platform
  in mind. The mapping to a platform is the lifecycle builder's durable
  output, not this file's.
- No trace of the builder: the deposited file never carries this skill's
  disposable marker, name, or paths.
- Governance is adjacent, not inlined: agent authority lives in
  `.agents/knowledge/agent-authority.md`, produced by `meta-agent-authority`.
  The contract may point to it; it must not restate it.
- Specifications are adjacent, not inlined: the specification discipline —
  level, tool, artifact map, approval gate — lives in
  `.agents/knowledge/spec-workflow.md`, produced by `meta-spec-workflow`.
  The contract may point to it; it must not restate it, and a
  specification is the content of a change request's acceptance, never a
  semantic of its own.
