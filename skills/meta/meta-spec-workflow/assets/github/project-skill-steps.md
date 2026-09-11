<!-- Slot texts for the GitHub base's project skill
(`github-project-workflow`). Insert each block at the step the platform
builder's durable-harness reference registers, as the table in
references/github-expression.md directs. Rework every <angle-bracket>
value from the contract; choose one alternative where two are offered;
leave no placeholder and no comment behind. -->

## TAKE_WORK_PRECONDITION — `## Take work` step 1, after "acceptance criteria are executable."

Combined shape: An issue with no change record yet is taken by committing
the record to the draft PR first (step 3) and stopping there <discussion-closed: "until the gate owner closes the discussion on the draft" | blocking: "until the gate owner's `<exact text>` comment exists">; the record's scenarios are then the acceptance criteria.

Split shape: An issue whose specification PR is not merged is escalated to
the gate owner, not executed; the merged record's scenarios are the
acceptance criteria.

## DRAFT_FIRST_CONTENT — `## Take work` step 3, after "the claim and the work log."

Combined shape only: Its first push is the change record — the proposal
and the delta specs, created through the spec tool's commands and passing
its validator, with no design or tasks — and the body's `Phase:` line
reads `specification` while Changes and Validation keep their reserved
line. Then stop. <discussion-closed: "The gate owner discusses on the PR
and directs record changes in conversation; push each through the publish
gate. When the gate owner says in conversation that the discussion is
closed, read the PR's comments (`gh api repos/<owner/repo>/issues/<n>/comments`)
and its review threads with their resolution state (the GraphQL
`reviewThreads` connection, field `isResolved`); list every unresolved
thread, every adjustment requested in the discussion that the record does
not carry, and every pair of conclusions that contradict each other; ask
the gate owner to confirm them; and start design, tasks, and
implementation only when nothing is open or the open items are confirmed.
Record the closing on the `Approval:` line." | blocking: "Wait for the gate
owner's `<exact text>` comment; it covers the record as of the last push
before it, and a later push to the record needs a fresh comment unless the
gate owner decided that push in conversation. Then run the same
reconciliation of comments and review threads before design.">

## CREATE_WORK_RULE — `## Create issues`, before the tracking-issue sentence

An issue opens when the requirement appears, carrying the raw requirement,
owner, and priority and no acceptance criteria; it links the change record
once that exists. Issues derived from the record's task list are optional,
one per task that independently earns state, each linking the scenarios it
closes. Never copy acceptance criteria into an issue. <Split shape: A
specification PR references its issue with `Refs #N`, never `Closes`.>

## FINISH_STEP — `## Finish` step 2, after "update the final description."

Set the `Phase:` line to `implementation`; replace the reserved line of
Changes with permalinks to the commits (the exact lines for a local
change, the whole file or directory for a broad one) and the reserved line
of Validation with each scenario and its result, linking the plan; and
confirm the spec-side step: <in-request: "the change record is archived in
this PR before it is marked ready" | automated: "every task of the change
record is ticked so the `<archive workflow name>` workflow archives it
after merge">. The local check command runs the specification validator; a
red validator is a red check.
