# Specifications in Tracked Work

Read before asking questions 3–8 of step 2 and again in step 7. It is the
design basis for the change request shape, the approval mode, the archive
mode, the default author, and the specification scope, and for what the
second phase inserts into the platform base. Everything here is stated in
tool-neutral and platform-neutral terms; the deposited contract and the
inserted slot text use the platform's own words (issue or work item, pull
or merge request, workflow or job).

## The two change request shapes

| | Combined | Split |
|---|---|---|
| What carries the change record | The one change request, opened as a draft when the record is committed | A specification change request carrying only the record |
| Where the approval gate is exercised | On the draft, in the contract's approval mode | By approving and merging the specification change request |
| What follows approval | Reconciliation, then plan and tasks, then implementation on the same branch | One or more implementation change requests, each linking the merged record |
| Ready means | Implementation review of scenarios against the result | The same, per implementation change request |
| Archive or converge | Per the archive mode, inside the request before ready or after merge by automation | After the last implementation change request completes the tasks |
| What the integration branch may hold | Under in-request archiving, never an unarchived record | Approved records awaiting implementation, each owned by an open work item |
| Selecting fact | No consumer depends on a stable contract; feature-driven products; local or copied change propagation | Consumers depend on a stable contract: libraries, frameworks, shared infrastructure, service APIs; dependency or inherited change propagation; a standards or specification project |

Under split, a record on the integration branch that is approved but has no
open work item owning its implementation is stale: assign it or remove it
through a change request, never by hand. A specification change request
references the work item and does not close it; the final implementation
change request closes it. Its title follows the project's commit
convention for specification changes.

## Approval modes

| | Discussion-closed (default) | Blocking comment |
|---|---|---|
| What the gate owner does | Discusses on the draft, directs record changes in conversation, and declares the discussion closed in conversation | Posts a fixed-wording comment on the draft (for example `Specification approved`) |
| The record of approval | The closing instruction plus the request's discussion state at that moment | The comment; it covers the record as of the last push before it |
| A later push to the record | Nothing more: the gate owner directed it | A fresh comment, unless the gate owner decided that push in conversation (a narrowing, a dropped scenario) |
| What the agent does next | Reconciliation, then design and tasks | The same reconciliation, then design and tasks |
| Selecting fact | Default: the record is discussed where it is read, and the closing is a decision the gate owner states | The user or the project wants an explicit token on the request |

Reconciliation: the agent reads the request's comments and review threads
with their resolution state; lists every unresolved thread, every
adjustment requested in the discussion that the record does not yet carry,
and every pair of conclusions that contradict each other; asks the gate
owner to confirm the open items; and starts design, tasks, and
implementation only when nothing is open or the open items are confirmed.
The closing state is recorded on the request's approval line.

Neither mode uses the platform's review-approval state: GitLab removes
approvals when commits are added by default, GitHub does wherever a
ruleset dismisses stale approvals, and the approval would point at a tip
the implementation pushes replace. Under split, merging the specification
change request is the approval in both modes.

## Timing

| Step | Event | Who | Recorded where | Done when |
|---|---|---|---|---|
| 1 | Requirement appears (meeting, discussion, request) | The requester | A work item with the raw requirement, owner, priority; no acceptance criteria (an acceptance sketch is marked non-authoritative) | The item exists and links nothing yet |
| 2 | Specification written and clarified | The implementer by default, or a named planning role | The change record committed on a branch through the tool's commands; the draft (combined) or specification change request (split) published at once, then the agent stops | No clarification marker remains; the validator passes; the draft is public |
| 3 | Specification review — the approval gate | The gate owner | The closing of the discussion or the fixed comment on the draft; or the merge of the specification change request | The approval is recorded and the reconciliation found nothing open |
| 4 | Plan and tasks | The implementer | The change record | Every requirement maps to a decision; every scenario has a task |
| 5 | Implementation | The implementer | Commits; a deviation updates the specification and returns to step 3 | Every task closed or its deviation approved |
| 6 | Verification, then ready | The implementer, then the reviewer or a reviewing agent | The change request's validation section names each scenario and its result and links the plan | Every scenario passed or recorded as a spec change |
| 7 | Archive or converge | Per the archive mode | The archive directory and the source-of-truth spec | The spec and the code describe the same system |
| 8 | Merge | The integration decision owner | The closing keyword closes the work item | The item is closed by the merge, not by hand |

## The change request body

Under a specification discipline the request body navigates to the record
and carries no implementation until ready: an opening paragraph stating
the goal (what is true once it merges, not the work); a section stating
the value; a specification block — the record's path as a link on the
branch, the phase (specification or implementation), one link per record
file, and the approval line in the contract's mode (what closes it, or the
exact comment text on its own line so it can be copied); related work with
the closing reference; changes and validation reserved until ready —
changes as permalinks to the commits (the exact lines for a local change,
the whole file or directory for a broad one), validation naming each
scenario with its result and linking the plan. Further sections are
allowed and pass the publish gate like every other.

## Archive modes

**Automated.** An automation job on the integration branch, triggered by
each merge, archives every change whose task list has at least one
completed task and no open task, merges its delta into the source-of-truth
spec, validates strictly, commits, and pushes. Its rules:

- Serialized: at most one run at a time (a concurrency group or resource
  group on the job), never cancelling a run in progress.
- Idempotent: every run rescans all completed changes, not only the one
  the triggering merge carried.
- No retry: when the push is rejected because the branch moved, the run
  fails; the run that the competing merge triggered archives the rest.
- Self-validating: the job runs the strict validator itself, because a push
  by some identities triggers no further automation.
- Authorized: the automation identity needs permission to push to the
  protected integration branch — a platform setting a maintainer grants,
  recorded in the harness as a maintainer action; until it exists, the
  in-request mode is in force. The path differs by platform and by who
  owns the repository: the expression references record it.
- Committed under the project's commit convention.
- The integration branch briefly holds a completed, unarchived record
  between merge and the run; the harness says so.

**In-request.** The change request archives before it is marked ready, so
the integration branch never holds an unarchived change. Choose it when no
automation exists or the automation cannot push. It freezes the record
before review: a defect review finds in an archived record goes to the
request's validation section or a follow-up change, never into the
archived record. Prefer automated wherever a push path exists; the trade is
a briefly unarchived record against a frozen one under review.

The bundled `scripts/archive_completed_changes.py` implements the OpenSpec
case, including changes marked spec-less; the second phase wraps it in the
platform's workflow or job. Other tools have no fixed archive operation:
Spec-Kit archives nothing (a spec-first feature is complete when delivered;
a spec-anchored project's rule is what updates the living spec), Kiro ticks
tasks in its own files, and committed documents merge by hand. For those,
define with the user the completion criterion (a ticked task list, a status
line, a label in the document) and the post-processing step before
designing any job, and build it from the same rules above.

## Specification scope

Specification domains describe the product the project delivers. A change
to the project's own harness, tooling, checks, workflows, or documents is a
spec-less change: recorded with the tool's marker (OpenSpec: `skip_specs:
true`, with a proposal, design, and tasks and no delta spec), linked by
tracked work the same way, never given a domain. When a change removes or
reshapes a domain's capabilities, the domain's purpose line may need a hand
correction in the same request; the contract names that exception where
the tool otherwise forbids hand edits to main specs. When a change first
creates a domain and the schema requires a baseline block, its scenarios
are verified like any other; a failing baseline scenario for behavior the
change does not touch is filed as a separate defect and the record is
narrowed, never widened silently.

## Per-tool notes

- **OpenSpec.** A change is a directory (proposal, delta specs, optional
  design, tasks) created by the tool's command, never by hand. Its
  documentation supports archiving after merge or inside the pull request;
  the automated mode is the former made mechanical. Under split, approved
  records sit on the integration branch until their implementation
  requests land; a delta written against a domain spec that another change
  archived later may no longer apply, so re-validate at implementation
  start. Propose generates design and task files with the specification;
  the gate reviews the proposal and the delta specs only. The strict
  validator runs after every artifact edit, before publishing, before
  ready, and after archiving.
- **Spec-Kit.** The feature directory's specification merges in the
  specification change request under split; plan and tasks join the
  implementation request. There is no archive operation and no validator;
  the contract names the structural check the project adopts, if any.
- **Kiro.** The requirements file merges first under split; design and
  tasks join implementation. Ticks in the tasks file are status: under
  split, tick only inside the implementing change request. No CLI
  validator.
- **Committed documents.** The written rule names who merges the delta into
  the domain spec and when — inside the request before ready, or by the
  automation after merge — because nothing does it automatically; a
  required-headings lint is the validation substitute when the team will
  maintain one.
