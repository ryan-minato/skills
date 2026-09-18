<!-- Slot texts for the GitLab base's merge request template and task
template. Insert each block at the slot the platform builder's
durable-harness reference registers, as the table in
references/gitlab-expression.md directs. Rework every <angle-bracket>
value from the contract; choose one alternative where two are offered;
leave no placeholder and no comment behind. -->

## RELATED_WORK_LINES — merge request template, under `## Related work`

Spec: [<path of the change record>](<link to the record on the branch>)
Phase: specification
Records: <one link per file of the approval package — the specification and the design when warranted; the task list is listed as after-approval material>
Approval: <the gate that is open and what closes it — conversational: "package deliberation open on this draft — the gate owner reviews the complete package here and closes it in conversation; the task list and the implementation follow the reconciled package", later replaced by "implementation deliberation open — the record is frozen once the gate owner closes it" | recorded: the same line, followed by the mark to leave, in a fenced block holding exactly what the contract fixes>

## ACCEPTANCE_ITEM — inserted into the checklist item beginning "The change satisfies", before the final period

, or the scenarios of the linked change record

(the item then reads: `- [ ] The change satisfies the linked acceptance criteria, or the scenarios of the linked change record.`)

## CHECKLIST_ITEMS — between the acceptance item and "The documented local checks pass."

- [ ] <conversational: "The package deliberation was closed on this draft before the task list, and the implementation deliberation before the freeze" | recorded: "The gate owner's `<exact mark>` names the version each gate approved">, or this merge request carries the specification only.
- [ ] Every task of the change record is done and verified, or the specification is updated — the record is frozen after the implementation deliberation closes, and `<check job name>` stays red until it is.

## INTAKE_LINK_FIELD — task template, one section before `## Acceptance criteria`

## Specification

<!-- The path of the change record or specification this task implements. -->

## ACCEPTANCE_SOURCE — appended to the comment under `## Acceptance criteria`

, or a link to the scenarios of the linked change record. Never both.

## COMPLETION_SOURCE — the goal milestone's description, under `## Observable completion`

Link the specifications whose scenarios define the goal; do not restate them.
