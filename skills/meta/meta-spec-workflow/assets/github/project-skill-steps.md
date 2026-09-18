<!-- Slot texts for the GitHub base's project skill
(`github-project-workflow`). Insert each block at the step the platform
builder's durable-harness reference registers, as the table in
references/github-expression.md directs. Rework every <angle-bracket>
value from the contract; choose one alternative where two are offered;
leave no placeholder and no comment behind. -->

## TAKE_WORK_PRECONDITION — `## Take work` step 1, after "acceptance criteria are executable."

Combined shape: An issue with no change record yet is taken by committing
the record to the draft PR first (step 3) and stopping there until the gate owner closes the package deliberation <conversational: "in conversation" | recorded: "and leaves `<exact mark>` on the package's last commit">; the record's scenarios are then the acceptance criteria.

Split shape: An issue whose specification PR is not merged is escalated to
the gate owner, not executed; the merged record's scenarios are the
acceptance criteria.

## DRAFT_FIRST_CONTENT — `## Take work` step 3, after "the claim and the work log."

Split shape: When the issue has no merged specification PR yet, the draft
is the specification PR: its body references the issue with `Refs #N` in
place of `Closes #N`, its only content is the approval package (the
specification and the design when warranted, created through the spec
tool's commands and passing its validator), and its body reads `Phase:
specification`. Stop there; the
gate owner's approval and merge of that PR is the approval. Each
implementation PR links the merged record; only the last one carries
`Closes #N`.

Combined shape: The draft opens once the approval package is complete —
the specification and, when warranted, the design, created through the
spec tool's commands and passing its validator; a task list the tool
generated alongside them is pushed but marked as after-approval and kept
out of the review — and
the body's `Phase:` line reads `specification` while Changes and
Validation keep their reserved line. Then stop. The gate owner discusses on the PR
and directs record changes in conversation; push each through the publish
gate. When the gate owner closes the package deliberation <conversational:
"in conversation" | recorded: "and leaves `<exact mark>` on the package's
last commit">, read the PR's comments (`gh api repos/<owner/repo>/issues/<n>/comments`)
and its review threads with their resolution state (the GraphQL
`reviewThreads` connection, field `isResolved`); list every unresolved
thread, every adjustment requested in the discussion that the record does
not carry, and every pair of conclusions that contradict each other; ask
the gate owner to confirm them; and start the task list and the
implementation only when nothing is open or the open items are confirmed.
Record the closing on the `Approval:` line. The same reconciliation runs
again at the implementation deliberation, before the record is frozen.

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
confirm the spec-side step: every task of the change record is ticked,
and the record is frozen in this PR — with the tool's archive command, or the lock the framework skill names where the tool has none —
once the gate owner closes the deliberation on the finished
implementation. Marking the PR ready is what opens that deliberation, so
the `<check job name>` check the framework skill installed is red until
the archive commit lands: that red is the merge block, not a defect. A
commit after the archive commit spends the closing; say so and ask
again.
