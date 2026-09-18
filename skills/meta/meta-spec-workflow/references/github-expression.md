# Shaping the GitHub Base for the Contract

Read in step 7 when the evidenced platform is GitHub and the platform
builder has delivered its base. The base registers its extension slots in
its own durable-harness reference (`## Extension slots`): a slot is a
heading, a step, or a field id, never a marker. This file says what to
insert into each slot for the contract's shape, approval package and
the gate modes, and the freeze; the exact texts live in `assets/github/`.
Never re-decide a contract fact here.

## Fill contract

- Locate each slot by its structure — the heading, the step number, the
  field id the base's table names — never by searching for a placeholder.
- Insert; never reword base text. The security checklist item stays
  byte-identical: the checklist workflow keys on the word "secrets", so
  keep that word out of every inserted item.
- Before inserting, grep the target file for the first sentence of the
  text about to be inserted; when it is present, skip the slot. A second
  run on the same base changes nothing.
- Register one `SYNC_ROW` per insertion in the `## Synchronization` table
  of `.agents/knowledge/github-workflow.md`, pairing the inserted text with
  the contract fact it came from.
- After every slot: `grep -rn '{{[A-Z]'` over the delivered paths returns
  nothing (builder placeholders are uppercase names; the archive
  workflow's `${{ github.… }}` and `${{ secrets.… }}` expressions stay); every workflow parses; a body built from the shaped template
  passes the checklist workflow; a clean-context read of the project skill
  can state the take-work precondition, the draft's first content, the
  reconciliation, and the finish step.

## Slot table

| Slot | Base location | Insert (asset section) | When |
|---|---|---|---|
| `RELATED_WORK_LINES` | PR template, `## Related work`, after the closing-keyword comment | the specification block: `Spec:`, `Phase:`, records, `Approval:` in the contract's mode | always |
| `ACCEPTANCE_ITEM` | PR template, the checklist item beginning "Acceptance criteria" | ", or the scenarios of the linked change record" | always |
| `CHECKLIST_ITEMS` | PR template, between the acceptance item and the security item | the approval item worded for the mode and the package; the archive item worded for the archive executor | always |
| `INTAKE_LINK_FIELD` | task and feature forms, before the field with id `acceptance` | the optional `spec` input | always |
| `ACCEPTANCE_SOURCE` | task and feature forms, the `acceptance` field's `description` | the change-record alternative, "never both" | always |
| `COMPLETION_SOURCE` | tracking-issue body, `## Observable completion` | link the specifications whose scenarios define the goal | always |
| `TAKE_WORK_PRECONDITION` | project skill, `## Take work` step 1 | combined: record to the draft first, then stop; split: unmerged specification PR is escalated | always |
| `DRAFT_FIRST_CONTENT` | project skill, `## Take work` step 3 | combined: the record as first push, `Phase: specification`, reserved sections, stop; the wait and the reconciliation for the mode. Split: `Refs #N` in place of the base's `Closes #N` on the specification PR, the record as its only content, `Closes #N` on the last implementation PR | always |
| `CREATE_WORK_RULE` | project skill, `## Create issues` | issues carry no acceptance criteria and link the record; task-derived issues optional; `Refs #N` for a specification PR | always |
| `FINISH_STEP` | project skill, `## Finish` step 2 | `Phase: implementation`, the reserved lines replaced, the spec-side step for the archive executor, the check the framework skill installed | always |
| `KNOWLEDGE_SECTION` | `.agents/knowledge/github-workflow.md`, appended `## Specifications` | the contract's location, the slots filled because of it, the framework skill that owns the check, the commands, and the labels, the update trigger "when the spec directory or tool changes, re-check every template link" — never the contract's tables | always |
| `SYNC_ROW` | `.agents/knowledge/github-workflow.md`, the `## Synchronization` table | one row per insertion above | per insertion |
| `MAINTAINER_ACTION` | `platform-settings.md`, one row each | the actions the framework skill names, such as the label sync | the framework skill's automation |

The check, the comment commands, the status labels, and the validator's
place in the local check command are the framework skill's to install
(the handoff of step 3); this builder edits no workflow. None of them
archives: that is a person's command on the branch.
Do not add a "Specification" issue type: a spec is a document in the
repository, and its lifecycle lives in the tool's layout. The `spec/*`
status labels are facts a workflow derives from the record and are
applied by that workflow, never by hand. The bug report form is unchanged:
a bug's acceptance baseline is the expected behavior, which under a
spec-anchored contract is the spec's current requirement.

## Ready-state rules

State them in the knowledge section and, when the project keeps a body
check, give that check these rules: a draft may hold the reserved lines
and `Phase: specification`; a ready pull request needs `Phase:
implementation`, a `Spec:` line, no reserved line in Changes or
Validation, and every checklist item ticked. Do not require the
framework skill's check to be green here: it is red from the moment
the request is ready until the freeze commit lands, and that red is
the merge block for the deliberation in between. Green is a
condition of merging, never of becoming ready. The base's checklist workflow enforces headings and the security
item only; a body check that knows the phase is a project asset, not this
builder's.

## Bot identity

The framework skill's jobs read and comment with the platform token and
write nothing to the repository, so no push identity and no secret is
involved and the comment events they cause start no run. The framework
skill states the facts; this builder records only the maintainer actions
those need, such as the label sync.

## Platform-native option

When the contract records committed specification documents with no tool,
the same slots apply: the `spec` input holds a repository path, the
checklist carries the archive item, no framework skill exists, and
`checks.md` says the checklist is the only gate. Do not promote issues
or Discussions into the specification store — a closed issue reads as
"done", not as a requirement.
