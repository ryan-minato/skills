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
package) → approved (<the approval owner closing the discussion on the
draft in conversation, reconciled with nothing open | the approval owner's
`<exact text>` comment on the draft, covering the package as of the last
push before it>, or the merged specification <pull request | merge
request>) → implemented (the task list written after the closing, every
task done and every scenario verified, recorded in the <pull request |
merge request>'s validation section) → archived (delta merged into the
source-of-truth spec and the record moved to the archive inside the <pull
request | merge request> before ready, by the executor below).
Name the tool command category for each move without quoting the
command.>

## Approval gate

<Who approves a specification before planning and implementation start —
a role or a person, never "the team" — and whether an agent may approve a
specification it wrote. Agent authority levels are governed by
`.agents/knowledge/agent-authority.md`; this gate is where they attach.>

The gate is exercised on the approval package: the specification plus
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

<Discussion-closed: <the approval owner> discusses on the draft and
directs record changes in conversation; each change is pushed through the
publish gate. The approval is recorded when <the approval owner> says in
conversation that the discussion is closed: the agent then reads the
<pull request | merge request>'s comments and review <threads |
discussions> with their resolution state, lists every unresolved one,
every requested adjustment the record does not carry, and every pair of
contradicting conclusions, confirms them with <the approval owner>, and
starts the task list and the implementation only when nothing is open or
the open items are confirmed; the closing is noted on the draft's
`Approval:` line.
| Blocking: <the approval owner> posts the comment `<exact text>` on the
draft; it covers the record as of the last push before it, and a later
push to the record needs a fresh comment unless <the approval owner>
decided that push in conversation. The same reconciliation of comments
and review <threads | discussions> runs before the task list.> A platform review
approval is not the record in either mode, because later pushes dismiss
it.

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
before it is marked ready, so <default branch> never holds an unarchived
record; the archived record is frozen under review: a defect found in it
is recorded in the request's Validation section or a follow-up change,
never edited into the archive. Executor: <by hand — every task ticked,
then the tool's archive command, the validator, and a commit on the
branch; the request checklist is the gate | the automation the framework
skill `<framework skill name>` installed: the `<check job name>` check
fails a ready <pull request | merge request> that still holds an
unarchived record (a warning while it is a draft); the `<trigger label>`
label makes the `<archive job name>` <workflow | job> archive every
complete related record, commit, push to the branch, and remove the label
— applying the label is an authorized remote write, and <GitHub: after
the bot's push a user with write access approves the workflow runs the
push queued | GitLab: the label takes effect on the next pipeline>; a
<pull request | merge request> from a fork is archived by its author from
the commands the bot posts>. Selected because <the automation evidence
that decided it>.

## Request automation

<Delete when no framework skill is installed.> The framework skill
`<framework skill name>` owns the request automation and its script:
`<command syntax>` shows a related record's documents and
`<command syntax>` its task progress <on GitHub as comment commands | on
GitLab as manual jobs>; the `<labels job name>` <workflow | job> keeps the
status labels `<archived-axis labels>` and `<progress-axis labels>` on
every <pull request | merge request> from the related records' task lists
— read them, never set them by hand. Maintainer actions: <label sync | the
label creation and the token variables>; <the approval click after a bot
push | "pipelines must succeed">.

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
- The archive executor changes, or a <workflow | job>, command, or label of
  the request automation is added, renamed, or removed.
- The rule for when a design is warranted changes.
