<!--
Raw shape for the specification contract deposited into the target project
at .agents/knowledge/spec-workflow.md. Rework every line against the
settled answers, delete every section the design does not use, and remove
every placeholder and this comment before the file is written. Write the file in
the project's platform vocabulary: <angle-bracket> slots offer the GitHub
and GitLab objects to choose from; no design word such as "tracked work"
or "change request" may survive. Tool names are facts and stay.
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
| Behavior of a domain | `<source-of-truth spec>` | knowledge base, <issues | work items> |
| Acceptance of a change | the scenarios in `<change record>/…` | <pull requests | merge requests>, <issues | work items> |
| Conventions and mechanics | the knowledge base | specifications never restate them |

A file listed under "Points to it" may summarize in one line and must link;
it never restates the fact.

## Lifecycle

<States a specification passes through, the event that moves it, and where
that event is recorded, e.g. proposed (change record committed and the
draft <pull request | merge request> opened) → approved (the approval
owner's comment on the draft naming the commit, or the merged
specification <pull request | merge request>) → implemented (every task
done and every scenario verified, recorded in the <pull request | merge
request>'s validation section) → archived (delta merged into the
source-of-truth spec and the record moved to the archive, by the archive
job after merge or by the <pull request | merge request> before ready).
Name the tool command category for each move without quoting the
command.>

## Approval gate

<Who approves a specification before planning and implementation start —
a role or a person, never "the team" — and whether an agent may approve a
specification it wrote. Agent authority levels are governed by
`.agents/knowledge/agent-authority.md`; this gate is where they attach.>

The gate is exercised as soon as the specification is written and
clarified and the <draft pull request | draft merge request> carrying it
is open — before any plan or task list exists. It reviews the outcome
description, each item as this project needs it: goals and scope,
terminology and domain model, behavior, invariants, constraints and rules,
states and transitions, interface and data contracts, exceptions and edge
cases, security and permissions, metrics and acceptance criteria. It never
reviews design or tasks: those are the implementer's after approval and are
judged by implementation review. The approval is recorded as a comment by
<the approval owner> on the draft naming the approved commit; a platform
review approval is not the record, because later pushes dismiss it.

## Change request shape

<Combined | Split>. Selected because <the change propagation line of
`.agents/knowledge/<platform>-workflow.md`, or the consumer contract, that
decided it>.

<Combined: one <pull request | merge request> carries the whole lifecycle.
It opens as a draft the moment the change record is committed; the gate
is exercised on that draft; plan, tasks, and implementation follow the
recorded approval; marking it ready requests implementation review; the
merge closes the <issue | work item>. The default branch <never holds an
unarchived change record (in-request archiving) | holds a completed record
only between merge and the archive job (automated archiving)>.>

<Split: a specification <pull request | merge request> carries only the
change record, references the <issue | work item> without closing it, and
is discussed, approved, and merged; implementation <pull requests | merge
requests> link the merged record, and the last one closes the <issue |
work item>. The default branch holds approved records awaiting
implementation; a record with no open <issue | work item> owning its
implementation is stale — assign it or remove it through a <pull request |
merge request>, never by hand. A contract-level change in a combined-shape
project may take this path as a recorded deviation.>

Default specification author: <the implementer | role>. The author
publishes the draft; <the approval owner> approves it.

## Archive mode

<Automated | In-request>. Selected because <the automation evidence that
decided it>.

<Automated: after each merge to <default branch>, <the workflow | the job>
`<name>` archives every change record whose tasks are all complete —
one run at a time (<a concurrency group | a resource group> named
`<name>`, never cancelling a run in progress), rescanning every completed
record on each run, validating strictly, committing under the project's
commit convention, and failing without retry when its push is rejected
because the run the competing merge triggered archives the rest. The
automation identity's permission to push to <default branch> is a setting
<the maintainer> grants and records in
`<platform settings knowledge file>`; until it is granted, the in-request
mode below is in force.>

<In-request: the <pull request | merge request> archives its change record
before it is marked ready, so <default branch> never holds an unarchived
record.>

## Specifications and <issues | work items>

- A specification owns what is built, why, and its acceptance scenarios.
- An <issue | work item> is opened when the requirement appears, carrying
  the raw requirement, owner, and priority and no acceptance criteria; it
  links the change record once that record exists. It owns who does it,
  when, and its status. Acceptance criteria are never copied into it; an
  acceptance sketch is marked non-authoritative.
- <Issues | Work items> derived from the change record's task list are
  optional, each naming the scenarios it closes.
- A <pull request | merge request> names the change record it implements
  on a `Spec:` line, states its phase (specification or implementation),
  and states whether the specification was updated or archived; its
  description is the diff and never a requirement.
- Discussion of a specification in an <issue | work item> thread is
  deliberation; the record is the file at the approved commit.
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
- The automation identity's push authorization changes, or the archive job
  is added, renamed, or removed.
