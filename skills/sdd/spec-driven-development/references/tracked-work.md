# Running the Loop Against Tracked Work

Read when a step of the loop meets the project's tracked work: publishing
the draft, waiting for the approval, reconciling, drafting the change
request body, archiving before ready. Everything here is stated in
tool-neutral and platform-neutral terms; the project's contract, its
templates, and the framework skill use the platform's own words (issue,
pull request, merge request, workflow, job), and they win.

## What the contract answers

| Fact | Where it is recorded | Default when no contract exists |
|---|---|---|
| Level, tool, and framework skill | the contract's header and artifact map | the tool the project already runs; otherwise the family fitting the situation, offered to the user |
| Change request shape | the contract's shape section | combined |
| Approval owner and mode | the contract's approval gate | discussion-closed on the complete package: the gate owner closes the discussion in conversation and the agent reconciles before the task list |
| When a design is warranted | the contract's approval gate | more than one reasonable approach, or structure, interfaces, dependencies, or files outside the record are touched |
| Archive executor | the contract's archive section | by hand, inside the request, before ready |
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

| | Discussion-closed (default) | Blocking comment |
|---|---|---|
| What the gate owner does | Discusses on the draft, directs record changes in conversation, and declares the discussion closed in conversation | Posts the fixed-wording comment the contract names (for example `Specification approved`) on the draft |
| The record of approval | The closing instruction plus the request's discussion state at that moment | The comment; it covers the package as of the last push before it |
| A record change directed while waiting | Update the record, publish it through the project's publish gate, keep waiting | The same; a push the gate owner did not decide in conversation needs a fresh comment before the task list, a narrowing the gate owner decided does not |
| What follows | Reconciliation, then the task list | Reconciliation, then the task list |

A platform review approval is not the record in either mode, because
later pushes dismiss it and it then points at a tip the implementation
replaces. Under split, merging the specification change request is the
approval.

## The two shapes at run time

| Step | Combined | Split |
|---|---|---|
| Publish (loop step 3) | Commit the package; open a draft change request whose first content is the package; stop | Commit the package; open a specification change request carrying only the record; stop |
| Approval | In the contract's mode, on the draft | The specification change request is approved and merged |
| Tasks and implementation | On the same branch, after the approval | On one or more implementation change requests, each linking the merged record; re-validate the delta first — a domain spec another change archived since may have moved |
| Ready | Implementation review of the scenarios against the result, after the record is archived or converged | The same, per implementation change request |
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

When the gate owner says the discussion is closed (or the comment is
posted), before any task list:

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
   branch; `Phase: specification` until the approval, `implementation`
   after; one link per file of the approval package, with the task list
   listed as after-approval material; an `Approval:` line stating the
   mode's state — "discussion open on this draft" and what closes it, or
   the exact comment text on its own line so it can be copied.
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

## Archiving before ready

The record is archived (spec-first) or converged into the source-of-truth
spec (spec-anchored) inside the change request before it is marked ready,
so the integration branch never holds an unarchived record. The contract
names the executor:

- **By hand** (the default): every task ticked and verified, then the
  tool's archive command, then the validator, then a commit on the
  request's branch.
- **The automation the framework skill installed**: apply the trigger
  label the contract names — an authorized remote write — and wait for the
  bot's commit and summary; pull it; do not archive by hand in parallel.
  Where the platform holds the bot's checks for a human's approval, the
  summary says so and the request stays blocked until someone with write
  access approves them.
- **On a fork**: the bot pushes nothing; it posts the commands. Run them,
  validate, commit, push.

The archived record is frozen: a defect review finds afterwards goes to
the request's validation section or a follow-up change, never into the
archive. A change with an open task is never archived; a task is never
ticked before its verification ran.

## Commands and labels a request may carry

Where the framework skill installed them, a request answers comment
commands that show a related change's documents and its task progress,
and carries status labels the platform's automation derives from the
record — an archived-or-not axis and a not-started, in-progress, done
axis. Read them as facts; never apply or remove such labels by hand, and
never treat them as the record's source of truth. A required check fails
a ready request that still holds an unarchived related change and warns
while the request is a draft. The framework skill names the commands and
labels for its tool.
