# Running the Loop Against Tracked Work

Read when a step of the loop meets the project's tracked work: publishing
the draft, waiting for either gate, reconciling, drafting the change
request body, marking the request ready, freezing the record. Everything here is stated in
tool-neutral and platform-neutral terms; the project's contract, its
templates, and the framework skill use the platform's own words (issue,
pull request, merge request, workflow, job), and they win.

## What the contract answers

| Fact | Where it is recorded | Default when no contract exists |
|---|---|---|
| Level, tool, and framework skill | the contract's header and artifact map | the tool the project already runs; otherwise the family fitting the situation, offered to the user |
| Change request shape | the contract's shape section | combined |
| Approval owner and mode | the contract's approval gate | conversational at both gates: the gate owner closes each deliberation in conversation and the agent reconciles before acting |
| Whether an approval is recorded | the contract's approval gate | no recorded mechanism; add one only when someone outside the conversation must verify a named person accepted a named version |
| When a design is warranted | the contract's approval gate | more than one reasonable approach, or structure, interfaces, dependencies, or files outside the record are touched |
| Archive executor | the contract's archive section | a person on the request's branch, once the implementation deliberation closes |
| What tracked work links | the contract's tracked-work section | the record's path; acceptance criteria never copied |
| Specification scope | the contract's scope section | product domains; the project's own harness, tooling, checks, workflows, and documents are spec-less changes under the tool's marker |
| Request body | the project's request template | the default body below |

## Publishing and waiting

The draft opens once the approval package is complete: the specification,
clarified, plus the design when one is warranted, validated and committed.
Publish the package as the request's first content with the phase marker
"specification", then stop. Under the combined shape that is the draft
change request; under the split shape it is a specification change request
carrying only the record, referencing the work item without closing it,
titled per the project's commit convention for specification changes.

While waiting, write no task list and no code; say what is being waited
for, per the mode the contract records:

| | Conversational (default) | Recorded approval |
|---|---|---|
| What the gate owner does | Discusses on the draft, directs record changes in conversation, and declares the deliberation closed in conversation | The same, and then leaves the mark the contract names on the version being approved |
| What stands as the approval | The closing instruction plus the request's discussion state at that moment, and — at the second gate — the freeze commit | A durable, attributable mark naming one version: canonically the platform's review approval paired with the setting that dismisses it on a new commit; a fixed-wording comment where the platform has no such pair |
| A record change directed while waiting | Update the record, publish it through the project's publish gate, keep waiting | The same; the mark is spent when the version it named is replaced, so it is taken again after the last change |
| What follows | Reconciliation, then the next step | Reconciliation, then the next step |

**Recorded approval is a category, not one mechanism.** Choose it from
who reads the record: a conversational closing is legible only to the
people in the conversation, so it is enough until someone outside it —
an auditor, a release manager, a compliance reviewer, a downstream team
— must verify for themselves that a named person accepted a named
version. Team size and pipeline maturity do not decide this; a large
team whose reviewers are all in the conversation needs no mark, and a
solo maintainer shipping under an audit obligation does. Name the
property the mechanism must provide — durable, attributable, bound to
one version — check the target platform offers it, and decide per gate
rather than once for the project: many contracts want the second gate
recorded and the first one conversational.

Under split, merging the specification change request is the approval at
the first gate.

## The two shapes at run time

| Step | Combined | Split |
|---|---|---|
| Publish (loop step 3) | Commit the package; open a draft change request whose first content is the package; stop | Commit the package; open a specification change request carrying only the record; stop |
| Approval | In the contract's mode, on the draft | The specification change request is approved and merged |
| Tasks and implementation | On the same branch, after the approval | On one or more implementation change requests, each linking the merged record; re-validate the delta first — a domain spec another change archived since may have moved |
| Ready | Marked ready with the record still open, for the deliberation on the finished implementation; frozen once that closes | The same, per implementation change request — the specification change request merges with the record unfrozen, and is the only request that may |
| Merge | Closes the work item | The last implementation change request closes it |

Under split, a record on the integration branch that is approved but has
no open work item owning its implementation is stale: assign it or remove
it through a change request, never by hand.

## What the gate examines

The outcome description — goals and scope, terminology and the domain
model, behavior, invariants, constraints and rules, states and their
transitions, interface contracts, data contracts, exceptions and edge
cases, security and permissions, metrics and acceptance criteria — and,
when the package holds a design, its bounds: the approach, the technical
constraints, the preferences, the rejected alternatives. Never the task
list or a step breakdown. A design that reads as a numbered procedure is
rewritten as bounds before the draft opens; a task list the tool generated
is pushed as a draft, marked after-approval on the request, and kept out
of the review.

## Reconciliation checklist

When the gate owner says a deliberation is closed, before the task list
at the first gate and before the freeze at the second:

1. Read every comment on the request and every review thread with its
   resolution state (on GitHub: the issue comments endpoint and the
   `reviewThreads` connection with `isResolved`; on GitLab: the merge
   request discussions, where each note carries `resolvable` and
   `resolved`).
2. List every unresolved thread.
3. List every adjustment requested in the discussion that the package does
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
   branch; `Phase: specification` until the first gate closes,
   `implementation` after; one link per file of the approval package,
   with the task list listed as after-approval material; an `Approval:`
   line naming which gate is open and what closes it — and, where the
   contract records approvals, the exact mark to leave, on its own line
   so it can be copied.
4. Related work: the closing reference to the work item, or the reason
   none exists. A split-shape specification request uses a non-closing
   reference instead; the last implementation request carries the closing
   one.
5. Changes and Validation, each holding a reserved line until the request
   is marked ready. Then Changes lists every touched file as a permalink
   to the commit that changed it — the exact lines for a local change,
   the whole file or directory for a broad one — and Validation names
   each scenario with its result and links the design's verification
   plan for the cases and rubric instead of restating them.
6. The project's checklist.

Sections beyond these may be added when the change needs them; every
section, added or default, passes the project's publish gate before it is
published. Never paste the task list or a diff summary into a draft in
the specification phase.

## Ready, then the freeze

The request is marked ready while its record is still open, and the
deliberation on the finished implementation runs against that. The
project's specification check fails a ready request holding an
unarchived record, so the request is red for the whole deliberation;
that red is the merge block, and it is expected — never work around it,
and never freeze early to clear it.

Exactly one request is exempt, and only under the split shape: the
specification change request, which carries the record and implements
nothing. A project on the combined shape has no such request and no
exemption — a request that implemented nothing and still holds an
unfrozen record is simply unfinished, and the check says so. Tell the
check which shape the contract records; do not let a request through by
arguing that it did no work.

When the gate owner closes this deliberation, reconcile it, then freeze:
every task ticked and verified, the tool's archive command (spec-first)
or the write-back into the source-of-truth spec (spec-anchored), the
validator, a commit on the request's branch. The executor is a person
who holds the branch — its author, or a maintainer who pulled a fork's
branch. No job does this: no platform token can push to a fork, so such
a job could never serve an external contribution, and on a branch it
could reach it would be the only automation needing write access to the
repository's contents.

The frozen record stays frozen: a defect review finds afterwards goes to
the request's validation section or a follow-up change, never into the
archive. A change with an open task is never archived; a task is never
ticked before its verification ran.

Nothing revokes a closing made in conversation, so watch for a spent
one: a commit after the freeze commit means the approved version no
longer exists. Compare the branch tip with the freeze commit before
pushing, before marking anything approved, and before handing over; when
they differ, say so and ask for the gate again.

## Commands and labels a request may carry

Where the framework skill installed them, a request answers comment
commands that show a related change's documents and its task progress,
and carries status labels the platform's automation derives from the
record — an archived-or-not axis and a not-started, in-progress, done
axis. They exist for visibility: the state of a request is legible from
the list view without opening it, and the comment commands put the
record's text into the discussion thread where it is being discussed.
Neither drives the process. Read the labels as facts; never apply or
remove such a label by hand, and never treat one as the record's source
of truth. A required check fails a ready request that still holds an
unarchived related change and warns while the request is a draft. The
framework skill names the commands and labels for its tool.
