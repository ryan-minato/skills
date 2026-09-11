# Running the Loop Against Tracked Work

Read when a step of the loop meets the project's tracked work: publishing
the draft, waiting for the approval, drafting the change request body,
archiving. Everything here is stated in tool-neutral and platform-neutral
terms; the project's contract and templates use the platform's own words
(issue, pull request, merge request, workflow, job), and they win.

## What the contract answers

| Fact | Where it is recorded | Default when no contract exists |
|---|---|---|
| Level and tool | the contract's header and artifact map | the tool the project already runs; otherwise the family fitting the situation, offered to the user |
| Change request shape | the contract's shape section | combined |
| Approval owner and mode | the contract's approval gate | discussion-closed: the gate owner closes the discussion in conversation and the agent reconciles before designing |
| Archive mode | the contract's archive section | in-request |
| What tracked work links | the contract's tracked-work section | the record's path; acceptance criteria never copied |
| Specification scope | the contract's scope section | product domains; the project's own harness, tooling, checks, workflows, and documents are spec-less changes under the tool's marker |
| Request body | the project's request template | the default body below |

## The two shapes at run time

| Step | Combined | Split |
|---|---|---|
| Publish (loop step 2) | Commit the record; open a draft change request whose first content is the record, with the phase marker "specification"; stop | Commit the record; open a specification change request carrying only the record, referencing the work item without closing it, titled per the project's commit convention for specification changes; stop |
| Approval | In the contract's mode, on the draft (below) | The specification change request is approved and merged |
| Plan, tasks, implementation | On the same branch, after the approval | On one or more implementation change requests, each linking the merged record; re-validate the delta first — a domain spec another change archived since may have moved |
| Ready | Implementation review of the scenarios against the result | The same, per implementation change request |
| Merge | Closes the work item | The last implementation change request closes it |

Under split, a record on the integration branch that is approved but has
no open work item owning its implementation is stale: assign it or remove
it through a change request, never by hand.

## Approval modes

After publishing, do not write design, tasks, or code. Say what is being
waited for, per the mode the contract records:

| | Discussion-closed (default) | Blocking comment |
|---|---|---|
| What the gate owner does | Discusses on the draft, directs record changes in conversation, and declares the discussion closed in conversation | Posts the fixed-wording comment the contract names (for example `Specification approved`) on the draft |
| The record of approval | The closing instruction plus the request's discussion state at that moment | The comment; it covers the record as of the last push before it |
| A record change directed while waiting | Update the record, publish it through the project's publish gate, keep waiting | The same; a push the gate owner did not decide in conversation needs a fresh comment before planning, a narrowing the gate owner decided does not |
| What follows | Reconciliation, then plan | Reconciliation, then plan |

A platform review approval is not the record in either mode, because
later pushes dismiss it and it then points at a tip the implementation
replaces. Under split, merging the specification change request is the
approval.

## Reconciliation checklist

When the gate owner says the discussion is closed (or the comment is
posted), before any design:

1. Read every comment on the request and every review thread with its
   resolution state (on GitHub: the issue comments endpoint and the
   `reviewThreads` connection with `isResolved`; on GitLab: the merge
   request discussions with `resolved`).
2. List every unresolved thread.
3. List every adjustment requested in the discussion that the record does
   not carry; update the record or ask whether the request was withdrawn.
4. List every pair of conclusions that contradict each other.
5. Ask the gate owner to confirm the open items; proceed only when nothing
   is open or the open items are confirmed.
6. Record the closing state on the request's approval line.

## Default request body

Follow the project's template when one exists. Absent one, the body has:

1. An opening paragraph, no heading, stating the goal — what is true once
   the request merges, not the work done.
2. A section stating the value: why this is worth merging now.
3. A specification block: `Spec:` as a link to the change record on the
   branch; `Phase: specification` until the approval, `implementation`
   after; one link per record file (design and tasks appear when they
   exist); an `Approval:` line stating the mode's state — "discussion open
   on this draft" and what closes it, or the exact comment text on its own
   line so it can be copied.
4. Related work: the closing reference to the work item, or the reason
   none exists.
5. Changes and Validation, each holding a reserved line until the request
   is marked ready. Then Changes lists every touched file as a permalink
   to the commit that changed it — the exact lines for a local change,
   the whole file or directory for a broad one — and Validation names
   each scenario with its result and links the plan for the cases and
   rubric instead of restating them.
6. The project's checklist.

Sections beyond these may be added when the change needs them; every
section, added or default, passes the project's publish gate before it is
published. Never paste the task list or a diff summary into a draft in
the specification phase.

## Archive modes at run time

- **Automated.** Leave the completed change for the archive job: every
  task ticked, the delta validated. The job runs one at a time, rescans
  every completed change on each run, and fails without retry when its
  push is rejected — the run the competing merge triggered archives the
  rest. Until the harness records the job's push path, the in-request
  mode is in force.
- **In-request.** Archive the record inside the change request before it
  is marked ready, so the integration branch never holds an unarchived
  record.

## Per-tool loop notes

- **OpenSpec.** A change is a directory (proposal, delta specs, optional
  design, tasks). Propose generates design and task files with the
  specification; the gate reviews the proposal and the delta specs only.
  The tool archives after merge or inside the pull request; the automated
  mode is the former made mechanical.
- **Spec-Kit.** Under split, the feature directory's specification merges
  in the specification change request; plan and tasks join the
  implementation request. There is no archive operation: a spec-first
  feature is complete when delivered, and a spec-anchored project's rule
  is what updates the living spec.
- **Kiro.** Under split, the requirements file merges first; design and
  tasks join implementation. Ticks in the tasks file are status: tick
  only inside the implementing change request.
- **Committed documents.** The contract names who merges the delta into
  the domain spec and when, because nothing does it automatically.
