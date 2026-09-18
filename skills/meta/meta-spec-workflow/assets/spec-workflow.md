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
<an issue | a work item> from a specification, and before editing any file
under `<spec tool directory>`.

Level: <spec-first | spec-anchored | spec-as-source>. Selected because <one
sentence naming the project fact that decided it>. Obligation: <what must
happen to the specification after a change ships>.
Approach: <tool name, or "committed documents"> — layout and commands
verified from the tool's own help and documentation on <absolute date>.
Re-verify a command from the tool before running or documenting it; do not
rely on a remembered one.

## Artifact operations

Every operation the tool has a command for — initializing, creating a
change or feature record, validating, archiving — runs through that
command, verified from the tool's help first; the tool's directory tree
and generated files are never created by hand. Hand edits stop at the
requirement text itself. Validation: <the tool's validator `<name>` runs
in strict mode as part of `<local check command>` | the tool ships no
validator; a required-headings lint in `<local check command>` stands in
| the tool ships no validator and no programmatic check exists; the
checklist is the gate>. It runs after every artifact edit, before the
draft is published, before the <pull request | merge request> is marked
ready, and after archiving; a red check is a red check.

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
draft <pull request | merge request> opened with the complete approval
package) → approved (<the approval owner closing the package
deliberation in conversation, reconciled with nothing open | the
approval owner's recorded mark on the package's last commit>, or the
merged specification <pull request | merge request>) → implemented (the
task list written after the closing, every task done and every scenario
verified, recorded in the <pull request | merge request>'s validation
section, the request marked ready for the second deliberation) →
archived (once that deliberation closes: delta merged into the
source-of-truth spec and the record moved to the archive inside the
<pull request | merge request>, by the executor below; that commit is
the freeze the final approval names).
Name the tool command category for each move without quoting the
command.>

## Approval gates

<Who owns each gate — a role or a person, never "the team" — and whether
an agent may approve a record it wrote. Agent authority levels are
governed by `.agents/knowledge/agent-authority.md`; these gates are where
they attach.>

There are two. The **package gate** releases the task list and the
implementation; the **freeze gate** releases the freeze, and its approval
applies to the frozen version. The request is marked ready before the
freeze, so `<check job name>` is red for the whole second deliberation:
that red is the merge block, not a defect.

The package gate is exercised on the approval package: the specification plus
the design when one is warranted — <the project's rule, by default: more
than one reasonable approach exists, or the change touches structure,
interfaces, dependencies, or files outside the record; a wording change
inside one section needs none>. The <draft pull request | draft merge
request> opens once the package is complete, before any task list exists,
and the agent that published it stops there. The review covers the outcome
description, each item as this project needs it — goals and scope,
terminology and domain model, behavior, invariants, constraints and
rules, states and transitions, interface and data contracts, exceptions
and edge cases, security and permissions, metrics and acceptance criteria
— and the design's bounds: the approach, the technical constraints, the
preferences, the rejected alternatives. The design lists no steps; a
design written as a procedure is rewritten before the draft opens. It is
committed, so it carries no secret or private data. The gate never
reviews the task list: it is the implementer's after approval and is
judged by implementation review.

<Conversational: <the approval owner> discusses on the request and
directs record changes in conversation; each change is pushed through the
publish gate. A gate closes when <the approval owner> says in
conversation that the deliberation is closed: the agent then reads the
<pull request | merge request>'s comments and review <threads |
discussions> with their resolution state, lists every unresolved one,
every requested adjustment the record does not carry, and every pair of
contradicting conclusions, confirms them with <the approval owner>, and
proceeds only when nothing is open or the open items are confirmed; the
closing is noted on the `Approval:` line. Nothing revokes such a
closing, so the agent compares the branch tip with the freeze commit
before pushing or handing over and asks again when they differ.
| Recorded: the same discussion and the same reconciliation, and then
<the approval owner> leaves `<the mechanism: the platform's review
approval with new commits dismissing it, or the comment `<exact text>`>`
on the version being approved. It names that version only; when a later
push replaces it, the mark is taken again.> Recorded per gate:
<which gates, and why someone outside the conversation needs to verify
the acceptance | neither: everyone who needs the decision is in the
conversation>.

## <Pull request | Merge request> shape

<Combined | Split>. Selected because <the change propagation line of
`.agents/knowledge/<platform>-workflow.md`, or the consumer contract, that
decided it>.

<Combined: one <pull request | merge request> carries the whole lifecycle.
It opens as a draft once the approval package is committed, with the
package as its only content and the agent stopped; the gate is exercised
on that draft; the task list and the implementation follow the recorded
approval and the reconciliation; marking it ready requests implementation
review; the merge closes the <issue | work item>. The default branch never
holds an unarchived change record.>

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

## Archive executor

Every change record is archived inside its <pull request | merge request>
once the freeze gate closes, so <default branch> never holds an
unarchived record; the frozen record stays frozen: a defect found in it
is recorded in the request's Validation section or a follow-up change,
never edited into the archive.

Executor: a person who holds the branch — <the implementer | role>, or a
maintainer who pulled a fork's branch. Every task ticked, then the tool's
archive command, the validator, and a commit. No <workflow | job>
archives: no platform token can push to a fork, and on a branch one could
reach it would be the only automation needing write access to
<repository | project> contents, to save a command the implementer
already runs.

The `<check job name>` check fails a ready <pull request | merge request>
that still holds an unarchived record (a warning while it is a draft), so
the request is red until the freeze. A commit after the freeze commit
means the approved version no longer exists: say so and ask for the gate
again.

## Request automation

<Delete when no framework skill is installed.> The framework skill
`<framework skill name>` owns the request automation and its script:
`<command syntax>` shows a related record's documents and
`<command syntax>` its task progress <on GitHub as comment commands | on
GitLab as manual jobs>; the `<labels job name>` <workflow | job> keeps the
status labels `<archived-axis labels>` and `<progress-axis labels>` on
every <pull request | merge request> from the related records' task lists
— read them, never set them by hand. All of it is read-only: the commands
put a record into the discussion thread where it is being discussed, and
the labels make a request's state legible from the list view. Neither
drives the process. Maintainer actions: <label sync | the label creation
and the token variables>; <nothing else | "pipelines must succeed">.

## Specifications and <issues | work items>

- A specification owns what is built, why, and its acceptance scenarios.
- An <issue | work item> is opened when the requirement appears, carrying
  the raw requirement, owner, and priority and no acceptance criteria; it
  links the change record once that record exists. It owns who does it,
  when, and its status. Acceptance criteria are never copied into it; an
  acceptance sketch is marked non-authoritative.
- <Issues | Work items> derived from the change record's task list are
  optional, each naming the scenarios it closes.
- A <pull request | merge request>'s description navigates to the record
  and carries no implementation until ready: an opening paragraph stating
  the goal, a section stating the value, the specification block (`Spec:`
  linking the record on the branch, `Phase:` specification or
  implementation, one link per file of the approval package with the task
  list marked as after-approval, `Approval:` in the mode above), related work with the closing reference (<a split-shape
  specification <pull request | merge request> references the <issue |
  work item> with <`Refs #N` | a bare `#N`> and never closes it; the last
  implementation request does | delete under the combined shape>), and
  Changes and Validation on their reserved line until the request is marked
  ready —
  then Changes as permalinks to the commits (the exact lines for a local
  change, the whole file or directory for a broad one) and Validation
  naming each scenario with its result and linking the plan. Further
  sections are allowed and pass the publish gate like every other. The
  description is never a requirement.
- Discussion of a specification in an <issue | work item> thread is
  deliberation; the record is the file at the approved commit.
- <Any deviation the user chose, with its reason.>

## Scope of specifications

Specifications cover behavior a change touches, and specification domains
describe the product this project delivers. A change to the project's own
harness, tooling, checks, workflows, or documents is a spec-less change:
<the tool's marker, as the framework skill names it, with a proposal,
design, and tasks and no delta spec>,
linked by <issues | work items> the same way, never a domain. <Main specs
change only through archived changes, with one exception: a change that
removes or reshapes a domain's capabilities corrects that domain's purpose
line by hand in the same <pull request | merge request>, named in its
proposal.> Untouched code is described by `<codebase map location>`, which
is not normative; before executing tasks in a module, compare its recorded
commit with the current one and re-map when they differ.

## Update this file when

- The tool is upgraded, renamed, or replaced, or its layout moves.
- The level changes (for example a spec-first project starts keeping specs).
- The approval owner or the approval mode changes.
- A second source of truth for any fact in the table above appears — merge
  it back, do not let it stand.
- The management model or the authority policy changes in a way that touches
  acceptance.
- The archive executor or the freeze changes, or a <workflow | job>,
  command, or label of the request automation is added, renamed, or
  removed.
- A gate's mode changes, or a recorded approval is added to or dropped
  from one.
- The rule for when a design is warranted changes.
