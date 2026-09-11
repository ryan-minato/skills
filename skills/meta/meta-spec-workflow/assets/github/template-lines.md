<!-- Slot texts for the GitHub base's pull request template, issue forms,
and tracking-issue body. Insert each block at the slot the platform
builder's durable-harness reference registers, as the table in
references/github-expression.md directs. Rework every <angle-bracket>
value from the contract; choose one alternative where two are offered;
leave no placeholder and no comment behind. -->

## RELATED_WORK_LINES — pull request template, under `## Related work`

Spec: [<path of the change record>](<link to the record on the branch>)
Phase: specification
Records: <one link per record file — proposal and delta specs; design and tasks are added when they exist>
Approval: <discussion-closed: "discussion open on this draft — the gate owner closes it in conversation; design and implementation follow the reconciled record" | blocking: "pending — the gate owner's comment on this draft, covering the record as of the last push before it; to approve, post this comment on one line:" followed by a fenced block holding exactly the text the contract fixes, for example `Specification approved`>

## ACCEPTANCE_ITEM — inserted into the checklist item beginning "Acceptance criteria", before "are met"

, or the scenarios of the linked change record

(the item then reads: `- [ ] Acceptance criteria of the linked issue, or the scenarios of the linked change record, are met`)

## CHECKLIST_ITEMS — between the acceptance item and the security item

- [ ] <discussion-closed: "The change record's discussion was closed on this draft before implementation" | blocking: "The gate owner's `<exact text>` comment on this draft is later than the record's last push">, or this pull request carries the specification only
- [ ] <in-request: "The change record is archived in this pull request, or the specification is updated" | automated: "Every task of the change record is done and it is left for the `<archive workflow name>` workflow, or the specification is updated">

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
