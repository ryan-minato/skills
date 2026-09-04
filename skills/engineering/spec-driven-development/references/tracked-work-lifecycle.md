# Specifications in Tracked Work

Read when designing or repairing how tracked work, change requests,
templates, and the archive step carry the specification, and when the
harness builder is declined or absent and the agent must record the
lifecycle itself. Everything here is stated in tool-neutral and
platform-neutral terms; write the project's harness in the platform's own
words (issue, pull request, merge request, workflow, job).

## The two change request shapes

| | Combined | Split |
|---|---|---|
| What carries the change record | The one change request, opened as a draft when the record is committed | A specification change request carrying only the record |
| Where the approval gate is exercised | On the draft, as a comment naming the approved commit | By approving and merging the specification change request |
| What follows approval | Plan and tasks, then implementation on the same branch | One or more implementation change requests, each linking the merged record |
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

## Timing

| Step | Event | Who | Recorded where | Done when |
|---|---|---|---|---|
| 1 | Requirement appears (meeting, discussion, request) | The requester | A work item with the raw requirement, owner, priority; no acceptance criteria (an acceptance sketch is marked non-authoritative) | The item exists and links nothing yet |
| 2 | Specification written and clarified | The implementer by default, or a named planning role | The change record committed on a branch; the draft (combined) or specification change request (split) published at once | No clarification marker remains; the draft is public |
| 3 | Specification review — the approval gate | The gate owner | A comment on the draft naming the approved commit; or the merge of the specification change request | The approval is recorded |
| 4 | Plan and tasks | The implementer | The change record | Every requirement maps to a decision; every scenario has a task |
| 5 | Implementation | The implementer | Commits; a deviation updates the specification and returns to step 3 | Every task closed or its deviation approved |
| 6 | Verification, then ready | The implementer, then the reviewer or a reviewing agent | The change request's validation section links the scenarios that passed | Every scenario passed or recorded as a spec change |
| 7 | Archive or converge | Per the archive mode | The archive directory and the source-of-truth spec | The spec and the code describe the same system |
| 8 | Merge | The integration decision owner | The closing keyword closes the work item | The item is closed by the merge, not by hand |

The work item links the change request; the change request's description
states the diff and the phase (specification or implementation) and links
the change record's path; neither restates a requirement.

## What specification review examines

The approval gate judges the outcome description, each item as the project
needs it: goals and scope; terminology and the domain model; behavior;
invariants; constraints and rules; states and their transitions; interface
contracts; data contracts; exceptions and edge cases; security and
permissions; metrics and acceptance criteria. Tasks and design are not its
object: they are how the outcome is built, belong to the implementer after
approval, and are judged by implementation review. A tool that generates
design and task files with the specification has produced drafts, not
review material.

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
- Self-validating: a push by the automation identity may trigger no further
  automation on some hosts, so the job runs the strict validator itself.
- Authorized: the automation identity needs permission to push to the
  protected integration branch — a platform setting a maintainer grants,
  recorded in the harness as a maintainer action; until it exists, the
  in-request mode is in force.
- Committed under the project's commit convention.
- The integration branch briefly holds a completed, unarchived record
  between merge and the run; the harness says so.

The bundled `scripts/archive_completed_changes.py` implements the OpenSpec
case; the platform builders wrap it in the host's workflow or job. Other
tools have no fixed archive operation: Spec-Kit archives nothing (a
spec-first feature is complete when delivered; a spec-anchored project's
rule is what updates the living spec), Kiro ticks tasks in its own files,
and committed documents merge by hand. For those, define with the user the
completion criterion (a ticked task list, a status line, a label in the
document) and the post-processing step before designing any job, and build
it from the same rules above.

**In-request.** The change request archives before it is marked ready, so
the integration branch never holds an unarchived change. Choose it when no
automation exists or the automation cannot push.

## Template lines the harness needs

Whatever the platform, the intake and change-request templates need only
these lines under a specification discipline:

- Work item: the raw requirement, owner, priority; a `Specification` field
  holding the change record's path once it exists; acceptance either
  executable criteria or a link to the record's scenarios — never both, and
  never a paste.
- Change request: a `Spec:` line naming the change record's path; a phase
  marker (specification or implementation); a checklist item "the change
  record was approved by the gate owner before implementation, or this
  request carries the specification only"; a checklist item "the
  specification was updated, or the change record was archived, or every
  task is complete and archiving runs after merge"; a validation section
  that links the scenarios that passed.
- Knowledge base: the change request shape, the archive mode, the gate
  owner, the default specification author, and the link-never-restate
  rule, in the platform's words.

Adding these lines to templates the project already has is within reach
without a builder; building forms, checks, automation, or a project skill
is the builder's work and is listed as remaining.

## Per-tool notes

- **OpenSpec.** A change is a directory (proposal, delta specs, optional
  design, tasks). Its documentation supports archiving after merge or
  inside the pull request; the automated mode is the former made
  mechanical. Under split, approved records sit on the integration branch
  until their implementation requests land; a delta written against a
  domain spec that another change archived later may no longer apply, so
  re-validate at implementation start. Propose generates design and task
  files with the specification; the gate reviews the proposal and the
  delta specs only.
- **Spec-Kit.** The feature directory's specification merges in the
  specification change request under split; plan and tasks join the
  implementation request. There is no archive operation.
- **Kiro.** The requirements file merges first under split; design and
  tasks join implementation. Ticks in the tasks file are status: under
  split, tick only inside the implementing change request.
- **Committed documents.** The written rule names who merges the delta into
  the domain spec and when — inside the request before ready, or by the
  automation after merge — because nothing does it automatically.
