# Designing the Contract

Read before asking questions 3–8 of step 2 and again in step 7. It is the
design basis for the change request shape, the approval package and mode,
the archive executor, the default author, and the specification scope,
and for what the second phase inserts into the platform base. Everything
here is stated in tool-neutral and platform-neutral terms; the deposited
contract and the inserted slot text use the platform's own words (issue
or work item, pull or merge request, workflow or job), and the framework
skill for the tool supplies every tool-specific fact.

## The two change request shapes

| | Combined | Split |
|---|---|---|
| What carries the change record | The one change request, opened as a draft when the approval package is committed | A specification change request carrying only the record |
| Where the approval gate is exercised | On the draft, in the contract's approval mode | By approving and merging the specification change request |
| What follows approval | Reconciliation, then the task list, then implementation on the same branch | One or more implementation change requests, each linking the merged record |
| Ready means | Implementation review of scenarios against the result, after the record is archived or converged inside the request | The same, per implementation change request |
| What the integration branch may hold | Never an unarchived record | Approved records awaiting implementation, each owned by an open work item |
| Selecting fact | No consumer depends on a stable contract; feature-driven products; local or copied change propagation | Consumers depend on a stable contract: libraries, frameworks, shared infrastructure, service APIs; dependency or inherited change propagation; a standards or specification project |

Under split, a record on the integration branch that is approved but has no
open work item owning its implementation is stale: assign it or remove it
through a change request, never by hand. A specification change request
references the work item and does not close it; the final implementation
change request closes it. Its title follows the project's commit
convention for specification changes.

## The approval package

The gate is exercised on the specification plus the design when one is
warranted. The default rule for "warranted": more than one reasonable
approach exists, or the change touches structure, interfaces,
dependencies, or files outside the record; a wording change inside one
section needs none; a project may fix the rule in its schema or contract
(for example: always, for a change that adds a file). The design is a
broad record of the chosen approach, the technical constraints, the
preferences, and the rejected alternatives — bounds the implementer works
inside, never a step list; a design written as a procedure is rewritten
before the draft opens. It is committed and published like every record,
so it carries no secret, credential, internal hostname, or private data.
The task list always follows approval, even when the tool generated it
with the specification. The framework skill says which files make up the
package for its tool.

## Approval modes

| | Discussion-closed (default) | Blocking comment |
|---|---|---|
| What the gate owner does | Discusses on the draft, directs record changes in conversation, and declares the discussion closed in conversation | Posts a fixed-wording comment on the draft (for example `Specification approved`) |
| The record of approval | The closing instruction plus the request's discussion state at that moment | The comment; it covers the package as of the last push before it |
| A later push to the package | Nothing more: the gate owner directed it | A fresh comment, unless the gate owner decided that push in conversation (a narrowing, a dropped scenario) |
| What the agent does next | Reconciliation, then the task list | The same reconciliation, then the task list |
| Selecting fact | Default: the package is discussed where it is read, and the closing is a decision the gate owner states | The user or the project wants an explicit token on the request |

Reconciliation: the agent reads the request's comments and review threads
with their resolution state; lists every unresolved thread, every
adjustment requested in the discussion that the package does not yet
carry, and every pair of conclusions that contradict each other; asks the
gate owner to confirm the open items; and starts the task list and the
implementation only when nothing is open or the open items are confirmed.
The closing state is recorded on the request's approval line.

Neither mode uses the platform's review-approval state: GitLab removes
approvals when commits are added by default, GitHub does wherever a
ruleset dismisses stale approvals, and the approval would point at a tip
the implementation pushes replace. Under split, merging the specification
change request is the approval in both modes.

The review covers the outcome description — goals and scope, terminology
and the domain model, behavior, invariants, constraints and rules, states
and transitions, interface and data contracts, exceptions and edge cases,
security and permissions, metrics and acceptance criteria — and the
design's bounds. It never covers the task list.

## Timing

| Step | Event | Who | Recorded where | Done when |
|---|---|---|---|---|
| 1 | Requirement appears (meeting, discussion, request) | The requester | A work item with the raw requirement, owner, priority; no acceptance criteria (an acceptance sketch is marked non-authoritative) | The item exists and links nothing yet |
| 2 | Specification written and clarified, design written when warranted | The implementer by default, or a named planning role | The change record committed on a branch through the tool's commands; the draft (combined) or specification change request (split) published with the complete package, then the agent stops | No clarification marker remains; the validator passes; the package is complete and public |
| 3 | Approval — the gate | The gate owner | The closing of the discussion or the fixed comment on the draft; or the merge of the specification change request | The approval is recorded and the reconciliation found nothing open |
| 4 | Task list | The implementer | The change record | Every scenario has a task and every task names its scenarios |
| 5 | Implementation | The implementer | Commits; a deviation updates the package and returns to step 3 | Every task closed or its deviation approved |
| 6 | Verification | The implementer, then the reviewer or a reviewing agent | The change request's validation section names each scenario and its result and links the design's verification plan | Every scenario passed or recorded as a spec change |
| 7 | Archive or converge, inside the request | The executor the contract names | The archive directory and the source-of-truth spec, committed on the request's branch | The spec and the code describe the same system; the request may be marked ready |
| 8 | Merge | The integration decision owner | The closing keyword closes the work item | The item is closed by the merge, not by hand |

## The change request body

Under a specification discipline the request body navigates to the record
and carries no implementation until ready: an opening paragraph stating
the goal (what is true once it merges, not the work); a section stating
the value; a specification block — the record's path as a link on the
branch, the phase (specification or implementation), one link per file of
the approval package with the task list marked as after-approval, and the
approval line in the contract's mode (what closes it, or the exact
comment text on its own line so it can be copied); related work with the
closing reference — a non-closing reference on a split-shape
specification request, whose last implementation request closes; changes
and validation reserved until ready — changes as permalinks to the
commits (the exact lines for a local change, the whole file or directory
for a broad one), validation naming each scenario with its result and
linking the design's verification plan. Further sections are allowed and
pass the publish gate like every other.

## Archiving

Every change record is archived (spec-first) or converged into the
source-of-truth spec (spec-anchored) inside its change request before the
request is marked ready, so the integration branch never holds an
unarchived record. The archived record is frozen under review: a defect
review finds afterwards goes to the request's validation section or a
follow-up change, never into the archive. A record with an open task is
never archived.

The contract names the **executor**:

- **By hand** — every task ticked, then the tool's archive command, the
  validator, and a commit on the request's branch; the request checklist
  is the gate. The default when no automation exists.
- **The framework skill's automation** — recommended when the platform
  runs CI. It consists of: a required check that fails a ready request
  holding an unarchived related record (a warning while the request is a
  draft) and fails a push to the integration branch that carries one;
  comment commands (or manual jobs) that show a related record's documents
  and its task progress; a trigger label whose bot archives every complete
  related record, commits, pushes to the request's own branch, and removes
  the label; and status labels on two axes — archived or not, and
  not-started, in-progress, or done — derived from the related records'
  task lists and applied by the automation, never by hand. A request from
  a fork is archived by its author from the commands the bot posts. Where
  the platform holds the bot's checks for a human's approval, that click
  is a recorded maintainer action. The framework skill installs all of it
  and names the commands, labels, jobs, and tokens; the contract records
  them as facts.

The rejected alternative — a job that archives after the merge by pushing
to the integration branch — needs a push identity with a protected-branch
bypass that some repositories cannot grant and leaves the integration
branch holding an unarchived record between the merge and the run. It is
not offered.

## Specification scope

Specification domains describe the product the project delivers. A change
to the project's own harness, tooling, checks, workflows, or documents is a
spec-less change: recorded with the tool's marker (the framework skill
names it), carrying a proposal, a design, and tasks and no delta spec,
linked by tracked work the same way, never given a domain. When a change
removes or reshapes a domain's capabilities, the domain's purpose line may
need a hand correction in the same request; the contract names that
exception where the tool otherwise forbids hand edits to main specs. When
a change first creates a domain and the schema requires a baseline block,
its scenarios are verified like any other; a failing baseline scenario for
behavior the change does not touch is filed as a separate defect and the
record is narrowed, never widened silently.

## Tools without an archive operation

A spec-first kit archives nothing: a feature is complete when every task
is ticked, and a spec-anchored project's rule is what updates the living
specification. An IDE's native spec files tick tasks in their own files.
Committed documents merge the delta into the domain document by hand
inside the request before ready. For each, the contract records the
completion criterion in place of an archive command, and the framework
skill (where one exists) installs the check that enforces it.
