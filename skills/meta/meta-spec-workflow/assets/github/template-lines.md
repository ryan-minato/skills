<!-- Slot texts for the GitHub base's pull request template, issue forms,
and tracking-issue body. Insert each block at the slot the platform
builder's durable-harness reference registers, as the table in
references/github-expression.md directs. Rework every <angle-bracket>
value from the contract; choose one alternative where two are offered;
leave no placeholder and no comment behind. -->

## RELATED_WORK_LINES — pull request template, under `## Related work`

Spec: [<path of the change record>](<link to the record on the branch>)
Phase: specification
Records: <one link per file of the approval package — the specification and the design when warranted; the task list is listed as after-approval material>
Approval: <the gate that is open and what closes it — conversational: "package deliberation open on this draft — the gate owner reviews the complete package here and closes it in conversation; the task list and the implementation follow the reconciled package", later replaced by "implementation deliberation open — the record is frozen once the gate owner closes it" | recorded: the same line, followed by the mark to leave, in a fenced block holding exactly what the contract fixes>

## ACCEPTANCE_ITEM — inserted into the checklist item beginning "Acceptance criteria", before "are met"

, or the scenarios of the linked change record

(the item then reads: `- [ ] Acceptance criteria of the linked issue, or the scenarios of the linked change record, are met`)

## CHECKLIST_ITEMS — between the acceptance item and the security item

- [ ] <conversational: "The package deliberation was closed on this draft before the task list, and the implementation deliberation before the freeze" | recorded: "The gate owner's `<exact mark>` names the version each gate approved">, or this pull request carries the specification only
- [ ] Every task of the change record is done and the record is archived in this pull request after the implementation deliberation closed, with no commit since, or the specification is updated

Keep the word "secrets" out of these items: the checklist workflow keys the
security item on it.

## INTAKE_LINK_FIELD — task and feature forms, before the field with id `acceptance`

  - type: input
    id: spec
    attributes:
      label: Specification
      description: Path of the change record or specification this <task implements | request extends>, if one exists.
    validations:
      required: false

## ACCEPTANCE_SOURCE — appended to the `acceptance` field's `description`

Task form: ` — or the scenarios of the linked change record. Never both.`
Feature form: ` — into the specification when the project keeps one.`

## COMPLETION_SOURCE — tracking-issue body, `## Observable completion`

Link the specifications whose scenarios define the goal; do not restate them.
