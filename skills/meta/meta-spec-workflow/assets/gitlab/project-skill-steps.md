<!-- Slot texts for the GitLab base's project skill
(`gitlab-project-workflow`). Insert each block at the step the platform
builder's durable-harness reference registers, as the table in
references/gitlab-expression.md directs. Rework every <angle-bracket>
value from the contract; choose one alternative where two are offered;
leave no placeholder and no comment behind. -->

## TAKE_WORK_PRECONDITION — `## Take and execute work` step 1, after "Confirm the work item is open."

Combined shape: A work item with no change record yet is taken by
committing the record to the draft MR first (step 3) and stopping there
<discussion-closed: "until the gate owner closes the discussion on the draft" | blocking: "until the gate owner's `<exact text>` comment exists">; the record's scenarios are then the acceptance criteria.

Split shape: A work item whose specification MR is not merged is escalated
to the gate owner, not executed; the merged record's scenarios are the
acceptance criteria.

## DRAFT_FIRST_CONTENT — `## Take and execute work` step 3, after "apply the approved labels and milestone."

Combined shape only: Its first push is the change record — the proposal
and the delta specs, created through the spec tool's commands and passing
its validator, with no design or tasks — and the description's `Phase:`
line reads `specification` while Changes and Validation keep their
reserved line. Then stop. <discussion-closed: "The gate owner discusses on
the MR and directs record changes in conversation; push each through the
publish gate. When the gate owner says in conversation that the discussion
is closed, read the MR's notes and discussions with their resolved state
(the merge request discussions API, field `resolved` per discussion); list
every unresolved discussion, every adjustment requested that the record
does not carry, and every pair of conclusions that contradict each other;
ask the gate owner to confirm them; and start design, tasks, and
implementation only when nothing is open or the open items are confirmed.
Record the closing on the `Approval:` line." | blocking: "Wait for the gate
owner's `<exact text>` comment; it covers the record as of the last push
before it, and a later push to the record needs a fresh comment unless the
gate owner decided that push in conversation. Then run the same
reconciliation of notes and discussions before design.">

## CREATE_WORK_RULE — a `## Create work` section inserted before `## Publish gate`

## Create work

A work item opens when the requirement appears, carrying the raw
requirement, owner, and priority and no acceptance criteria; it links the
change record once that exists. Items derived from the record's task list
are optional, one per task that independently earns state, each linking
the scenarios it closes. Never copy acceptance criteria into a
description. <Split shape: A specification MR references its work item
with a non-closing reference, never the closing pattern.>

## FINISH_STEP — `## Take and execute work` step 6, after "required checks pass,"

set the `Phase:` line to `implementation`, replace the reserved line of
Changes with permalinks to the commits (the exact lines for a local
change, the whole file or directory for a broad one) and the reserved line
of Validation with each scenario and its result linking the plan, confirm
the spec-side step — <in-request: "the change record is archived in this
MR before the draft flag is removed" | automated: "every task of the change
record is ticked so the `<archive job name>` job archives it after
merge"> — and then
