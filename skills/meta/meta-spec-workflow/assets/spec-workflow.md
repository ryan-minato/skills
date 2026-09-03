<!--
Raw shape for the specification contract deposited into the target project
at .agents/knowledge/spec-workflow.md. Rework every line against the
settled answers, delete every section the design does not use, and remove
every placeholder and this comment before the file is written. The file
must stay hosting-platform-neutral: no platform product or object names
anywhere. Tool names are facts and may stay.
-->

# Specification Workflow Contract

Read this before starting a change that alters behavior, before creating
tracked work from a specification, and before editing any file under
`<spec tool directory>`.

Level: <spec-first | spec-anchored | spec-as-source>. Selected because <one
sentence naming the project fact that decided it>. Obligation: <what must
happen to the specification after a change ships>.
Approach: <tool name, or "committed documents"> — layout and commands
verified from the tool's own help and documentation on <absolute date>.
Re-verify a command from the tool before running or documenting it; do not
rely on a remembered one.

## Artifact map

| Path | Holds | Edited by |
|---|---|---|
| `<principles file, e.g. the constitution or steering file>` | project-wide engineering principles the tool applies at plan time | humans; agents propose changes through a change |
| `<source-of-truth spec location>` | the current behavior of each domain (spec-anchored) or the spec of each feature (spec-first) | the specify step of a change |
| `<change record location>` | one directory per change: proposal, delta or feature spec, design, tasks | the change's author |
| `<archive location>` | completed change records | the archive step only |
| `<codebase map location>` | as-built description of untouched code, with the commit it was mapped at | re-mapped when the drift gate fires |

## Source of truth

| Fact | Rules | Points to it |
|---|---|---|
| Project goals | `<goal document>` | principles file, this contract |
| Engineering principles | `<principles file>` | agent entrypoint |
| Behavior of a domain | `<source-of-truth spec>` | knowledge base, tracked work |
| Acceptance of a change | the scenarios in `<change record>/…` | change requests, tracked work |
| Conventions and mechanics | the knowledge base | specifications never restate them |

A file listed under "Points to it" may summarize in one line and must link;
it never restates the fact.

## Lifecycle

<States a specification passes through and the event that moves it, e.g.
proposed (change record created) → approved (approval gate passed) →
implemented (scenarios verified) → archived (delta merged into the
source-of-truth spec, record moved to the archive). Name the tool command
category for each move without quoting the command.>

## Approval gate

<Who approves a specification before planning and implementation start —
a role or a person, never "the team" — and whether an agent may approve a
specification it wrote. Agent authority levels are governed by
`.agents/knowledge/agent-authority.md`; this gate is where they attach.>

## Specifications and tracked work

- A specification owns what is built, why, and its acceptance scenarios.
- Tracked work owns who does it, when, and its status, and links the
  specification or change record it serves. Acceptance criteria are never
  copied into tracked work.
- Work items for planned work are created from the change record's task
  list, each naming the scenarios it closes.
- A change request names the change record it implements and states
  whether the specification was updated or archived.
- <Any deviation the user chose, with its reason.>

## Scope of specifications

Specifications cover behavior a change touches. Untouched code is described
by `<codebase map location>`, which is not normative; before executing
tasks in a module, compare its recorded commit with the current one and
re-map when they differ.

## Update this file when

- The tool is upgraded, renamed, or replaced, or its layout moves.
- The level changes (for example a spec-first project starts keeping specs).
- The approval owner changes.
- A second source of truth for any fact in the table above appears — merge
  it back, do not let it stand.
- The management model or the authority policy changes in a way that touches
  acceptance.
