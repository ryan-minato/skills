# Durable Output

Read on every build, before depositing the contract.

## Where the contract lives

Deposit the adapted contract at `.agents/knowledge/spec-workflow.md` in the
target project (create the directory if the project has no knowledge tree;
if the project keeps agent knowledge elsewhere, follow the existing
convention and record the path in the hand-off). This file is the single
source of truth for the specification discipline: the governance builder
reads it to place the approval gate, and the platform lifecycle builders
read it to map — never re-decide — how intake templates, change-request
templates, and the project workflow skill refer to specifications.

The contract is not a specification and holds no requirement. Requirements
live in the tool-owned spec artifacts the contract's artifact map points to.

## The entrypoint pointer

Add one pointer to the project's agent entrypoint (`AGENTS.md` or its
equivalent), event-triggered rather than always-read:

> Read `.agents/knowledge/spec-workflow.md` before starting a change that
> alters behavior, before creating tracked work from a specification, and
> before editing any file under `<spec tool directory>`.

Where the tool keeps a project-wide principles file (a constitution or
steering file), the tool loads it itself; add an entrypoint pointer only
when the project's agents run without the tool's commands. Do not paste
contract or spec content into the entrypoint; one source of truth per fact.

## What the contract must carry

The deposited file, adapted from the asset, must state:

- The level (spec-first, spec-anchored, spec-as-source) and the fact that
  selected it, with the maintenance obligation the level implies.
- The approach and tool, the date the layout and commands were verified
  from the tool's own help and documentation, and the instruction to
  re-verify commands rather than trust remembered ones.
- The artifact map: every path the tool owns, what it holds, and which
  audience edits it.
- The source-of-truth table: for each kind of fact — engineering
  principles, behavior of a domain, acceptance of a change, conventions and
  mechanics, project goals — the one file that rules and the files that
  point to it.
- The specification lifecycle (for example proposed, approved, implemented,
  archived) and the event that moves a spec between states.
- The approval gate: who approves a specification before planning and
  implementation, and whether an agent may approve its own.
- The division of labor with tracked work, phrased so an agent can apply it
  to a concrete work item and change request.
- The rule that specifications cover changed behavior only, and where the
  as-built description of untouched code lives.
- An "Update this file when" list naming the events that reopen the design.

## Survival rules

- Hosting-platform-neutral forever: the file must read correctly with no
  hosting platform in mind. The mapping to intake forms, templates, and
  platform objects is the lifecycle builder's durable output, not this
  file's. Tool names are facts and may appear.
- No trace of the builder: the deposited file never carries this skill's
  disposable marker, name, or paths.
- Governance and management are adjacent, not inlined: agent authority
  lives in `.agents/knowledge/agent-authority.md` and the management model
  in `.agents/knowledge/project-workflow.md`. The contract may point to
  them; it must not restate them.
- No requirement in the contract: a behavior statement belongs in a spec.
