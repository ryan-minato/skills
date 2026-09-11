# Durable Output

Read on every build, before depositing the contract.

## Where the contract lives

Deposit the adapted contract at `.agents/knowledge/spec-workflow.md` in the
target project (create the directory if the project has no knowledge tree;
if the project keeps agent knowledge elsewhere, follow the existing
convention and record the path in the hand-off). This file is the single
source of truth for the specification discipline: the governance builder
reads it to place the approval gate, the platform builder reads it only to
know a paradigm contract exists, and this builder's second phase reads it
to fill the platform base's extension slots — never re-deciding a fact.

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
- The artifact operations: which operations run through the tool's
  commands (initializing, creating a record, validating, archiving), that
  hand edits stop at requirement text, the validator with its strict mode
  and its place in the local check command and the four moments it runs
  (after each artifact edit, before publishing the draft, before ready,
  after archiving) — or, for a tool without one, the structural check the
  project adopts or the statement that none exists.
- The artifact map: every path the tool owns, what it holds, and which
  audience edits it.
- The source-of-truth table: for each kind of fact — engineering
  principles, behavior of a domain, acceptance of a change, conventions and
  mechanics, project goals — the one file that rules and the files that
  point to it.
- The specification lifecycle (for example proposed, approved, implemented,
  archived) and the event that moves a spec between states.
- The approval gate: who approves a specification before planning and
  implementation, whether an agent may approve its own, and the mode —
  discussion-closed (the agent stops after publishing, the owner closes
  the discussion in conversation, the agent reconciles threads and record
  before implementing) or blocking (the fixed comment text, what it
  covers, when a fresh one is needed) — with the reconciliation step in
  both.
- The division of labor with tracked work, phrased so an agent can apply it
  to a concrete work item and change request, including when the work item
  opens and what it may not carry.
- The change request shape (combined or split) with its selecting fact, the
  default specification author, where the approval is recorded, what the
  integration branch may hold, and the request body's lines: the
  specification block and the sections reserved until ready.
- The archive mode (automated or in-request) with its selecting fact and,
  for automated, the job's name, its serialization, idempotence, and
  no-retry rules, and the push path by platform and owner type recorded
  as a maintainer action; for in-request, the freeze that review works
  under.
- The specification scope: domains cover the product; the project's own
  harness, tooling, checks, workflows, and documents are spec-less changes
  under the tool's marker; the hand-corrected purpose line as the one
  exception to tool-only edits of main specs.
- The rule that specifications cover changed behavior only, and where the
  as-built description of untouched code lives.
- An "Update this file when" list naming the events that reopen the design.

## Survival rules

- Platform vocabulary only: the file names the platform's objects and
  operations (issues or work items, pull or merge requests, drafts, the
  archive workflow or job) and never a term only this builder defines; the
  step-5 check enforces it. Which intake fields, template lines, project
  skill steps, and checks exist because of this contract is the second
  phase's output, recorded in the platform knowledge file's Specifications
  section, not in this file. Tool names are facts and appear as they are.
- No trace of the builder: the deposited file never carries this skill's
  disposable marker, name, or paths.
- Governance and management are adjacent, not inlined: agent authority
  lives in `.agents/knowledge/agent-authority.md` and the management model
  in `.agents/knowledge/<platform>-workflow.md`. The contract may point to
  them; it must not restate them.
- No requirement in the contract: a behavior statement belongs in a spec.
