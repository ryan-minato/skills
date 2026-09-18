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

## The two gates

A change passes two deliberations, and each one has its own mode:

| Gate | What it examines | What it releases |
|---|---|---|
| The package gate | The specification, and the design when one is warranted | The task list and the implementation |
| The freeze gate | The finished implementation against that package | The freeze — the archive or the convergence — which the approval then names |

The request is marked ready *before* the freeze, so the specification
check is red for the whole second deliberation. That red is the merge
block, and it is expected; never freeze early to clear it.

The shape decides one thing about that check, so record it where the
framework skill can read it. Under **combined** no request merges with
an unfrozen record. Under **split** exactly one does — the specification
change request, which carries the record and implements nothing — so the
check is configured to admit an unfrozen record with no completed task
and to fail as soon as one is completed. A combined project gets no such
exception: a request that implemented nothing and still holds an
unfrozen record is unfinished, not exempt.

## Approval modes

| | Conversational (default) | Recorded approval |
|---|---|---|
| What the gate owner does | Discusses on the request, directs changes in conversation, and declares the deliberation closed in conversation | The same, and then leaves the mark the contract names on the version being approved |
| What stands as the approval | The closing instruction plus the request's discussion state at that moment, and — at the freeze gate — the freeze commit | A durable, attributable mark naming one version |
| A later push | Nothing more: the gate owner directed it. Nothing revokes the closing either, so the agent detects a spent one by comparing the tip with the freeze commit | The mark is spent when the version it named is replaced; take it again after the last change |
| What the agent does next | Reconciliation, then the next step | The same reconciliation, then the next step |
| Selecting fact | Default: everyone who needs the decision is in the conversation | Someone outside the conversation must verify for themselves that a named person accepted a named version |

**Recorded approval is a category, not one mechanism.** Its canonical
form is the platform's review approval paired with the setting that
dismisses approvals when new commits arrive — GitLab's default, GitHub's
"dismiss stale pull request approvals" — because the pair is what binds
the mark to one version. Where a platform offers no such pair, a
fixed-wording comment covering the record as of the last push before it
does the same job less cleanly. Name the property the mechanism must
provide (durable, attributable, bound to one version), check the target
platform offers it, then choose the mechanism.

Recommend from who reads the record, never from team size or pipeline
maturity. A conversational closing is legible only to the people in the
conversation; that is enough until an auditor, a release manager, a
compliance reviewer, or a downstream team must confirm the acceptance
without asking anyone. A large team whose reviewers are all in the
conversation needs no mark; a solo maintainer under an audit obligation
does. Decide per gate: many contracts want the freeze gate recorded and
the package gate conversational.

Reconciliation: the agent reads the request's comments and review threads
with their resolution state; lists every unresolved thread, every
adjustment requested in the discussion that the package does not yet
carry, and every pair of conclusions that contradict each other; asks the
gate owner to confirm the open items; and starts the task list and the
implementation only when nothing is open or the open items are confirmed.
The closing state is recorded on the request's approval line.

Under split, merging the specification change request is the approval at
the package gate in both modes.

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
source-of-truth spec (spec-anchored) inside its change request, once the
freeze gate closes, so the integration branch never holds an unarchived
record. The frozen record stays frozen: a defect review finds afterwards
goes to the request's validation section or a follow-up change, never
into the archive. A record with an open task is never archived.

The **executor is a person who holds the request's branch** — the
implementer, or a maintainer who pulled a fork's branch. Every task
ticked, then the tool's archive command, the validator, and a commit.
The contract records that, and how a spent freeze is detected: a commit
after the freeze commit means the approved version no longer exists.

No job archives, and this is not a matter of project size. No platform
token can push to a fork, so an archiving job could never serve an
external contribution; on a branch it could reach, it would be the only
installed automation needing write access to the repository's contents,
to save one command the implementer is already running. A job that
archives after the merge by pushing to the integration branch is worse
still: it needs a push identity with a protected-branch bypass that some
repositories cannot grant, and it leaves the integration branch holding
an unarchived record between the merge and the run. Neither is offered.

What the framework skill does install is read-only, and it exists for
visibility rather than for process: a required check that fails a ready
request holding an unarchived related record (a warning while the
request is a draft) and fails a push to the integration branch that
carries one; comment commands (or manual jobs) that print a related
record's documents and its task progress into the discussion thread;
and status labels on two axes — archived or not, and not-started,
in-progress, or done — derived from the related records' task lists and
applied by the automation, never by hand, so a reviewer reads a
request's state from the list view. The framework skill installs them
and names the commands, labels, jobs, and tokens; the contract records
them as facts, and records which ones the project took.

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
inside the request. For each, the contract records the completion
criterion in place of an archive command, and the framework skill (where
one exists) installs the check that enforces it.

Such a tool still needs a freeze, or the approval has no version to
name. Absent an artifact, the freeze is a **declaration** on the request
— "the deliberation closed with the record unchanged at `<commit>`" —
and the contract fixes its wording. A spent declaration is detected the
same way as a spent archive commit: by a later commit.
