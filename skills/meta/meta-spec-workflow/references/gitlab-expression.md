# Shaping the GitLab Base for the Contract

Read in step 7 when the evidenced platform is GitLab and the platform
builder has delivered its base. The base registers its extension slots in
its own durable-harness reference (`## Extension slots`): a slot is a
heading, a step, or a section, never a marker. This file says what to
insert into each slot for the contract's shape, approval package and
the gate modes, and the freeze; the exact texts live in `assets/gitlab/`.
Never re-decide a contract fact here.

## Fill contract

- Locate each slot by its structure — the heading, the step number, the
  section the base's table names — never by searching for a placeholder.
- Insert; never reword base text. The sensitivity-review checklist item
  stays byte-identical.
- Before inserting, grep the target file for the first sentence of the
  text about to be inserted; when it is present, skip the slot. A second
  run on the same base changes nothing.
- Register one `SYNC_ROW` per insertion in the `## Synchronization` table
  of `.agents/knowledge/gitlab-workflow.md`.
- After every slot: `grep -rn '{{[A-Z]'` over the delivered paths returns
  nothing; the CI file lints against the target instance; a clean-context
  read of the project skill can state the take-work precondition, the
  draft's first content, the reconciliation, and the finish step.

## Slot table

| Slot | Base location | Insert (asset section) | When |
|---|---|---|---|
| `RELATED_WORK_LINES` | MR template, `## Related work`, after the reference-syntax comment | the specification block: `Spec:`, `Phase:`, records, `Approval:` in the contract's mode | always |
| `ACCEPTANCE_ITEM` | MR template, the checklist item beginning "The change satisfies" | ", or the scenarios of the linked change record" | always |
| `CHECKLIST_ITEMS` | MR template, between the acceptance item and "The documented local checks pass" | the approval item worded for each gate's mode and the package; the archive item worded for the freeze | always |
| `INTAKE_LINK_FIELD` | task template, one section before `## Acceptance criteria` | `## Specification` with its comment | always |
| `ACCEPTANCE_SOURCE` | task template, the comment under `## Acceptance criteria` | the change-record alternative, "never both" | always |
| `COMPLETION_SOURCE` | the goal milestone's description, under `## Observable completion` | link the specifications whose scenarios define it | always |
| `TAKE_WORK_PRECONDITION` | project skill, `## Take and execute work` step 1 | combined: record to the draft first, then stop; split: unmerged specification MR is escalated | always |
| `DRAFT_FIRST_CONTENT` | project skill, `## Take and execute work` step 3 | combined: the record as first push, `Phase: specification`, reserved sections, stop; the wait and the reconciliation for the mode. Split: a non-closing reference in place of the closing pattern on the specification MR, the record as its only content, the closing pattern on the last implementation MR | always |
| `CREATE_WORK_RULE` | project skill, a `## Create work` section before `## Publish gate` | items carry no acceptance criteria and link the record; task-derived items optional; a non-closing reference for a specification MR | always |
| `FINISH_STEP` | project skill, `## Take and execute work` step 6 | `Phase: implementation`, the reserved lines replaced, the spec-side step for the freeze, the check the framework skill installed | always |
| `KNOWLEDGE_SECTION` | `.agents/knowledge/gitlab-workflow.md`, appended `## Specifications` | the contract's location, the slots filled because of it, the framework skill that owns the check, the manual jobs, and the labels, the update trigger "when the spec directory or tool changes, re-check every template link" — never the contract's tables | always |
| `SYNC_ROW` | `.agents/knowledge/gitlab-workflow.md`, the `## Synchronization` table | one row per insertion above | per insertion |
| `MAINTAINER_ACTION` | the platform-settings knowledge, one row each | the actions the framework skill names: the label creation, the token variables, "pipelines must succeed" | the framework skill's automation |

The check, the manual jobs, the status labels,
and the validator's place in the local check command are the framework
skill's to install (the handoff of step 3); this builder edits no
pipeline. Do not add a specification work-item type: a spec is a document
in the repository, and its lifecycle lives in the tool's layout. The
`spec/*` status labels are facts a job derives from the record and are
applied by that job, never by hand. The issue and incident templates are
unchanged.

## Ready-state rules

State them in the knowledge section: a draft may hold the reserved lines
and `Phase: specification`; removing the draft flag needs `Phase:
implementation`, a `Spec:` line, no reserved line in Changes or
Validation, every checklist item ticked, and the framework skill's check
green. A pipeline job that checks the description against these rules is
a project asset, not this builder's.

## Automation identity

When the framework skill installs an archive job, it pushes to the
request's own source branch with a project access token the maintainer
creates as a masked variable; the framework skill states what the token
needs and what GitLab does not offer (no pipeline on a note or a label
change). This builder records the token creation as a maintainer action
and nothing else.

## Platform-native option

When the contract records committed specification documents with no tool,
the same slots apply: `## Specification` holds a repository path, the
checklist carries the archive item, and no framework skill exists. Do not promote work items or the Wiki into the specification
store — a closed item reads as "done", not as a requirement.
