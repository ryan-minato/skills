# Shaping the GitLab Base for the Contract

Read in step 7 when the evidenced platform is GitLab and the platform
builder has delivered its base. The base registers its extension slots in
its own durable-harness reference (`## Extension slots`): a slot is a
heading, a step, or a section, never a marker. This file says what to
insert into each slot for the contract's shape, approval mode, and archive
mode; the exact texts live in `assets/gitlab/`. Never
re-decide a contract fact here.

## Fill contract

- Locate each slot by its structure — the heading, the step number, the
  section the base's table names — never by searching for a placeholder.
- Insert; never reword base text. The sensitivity-review checklist item
  stays byte-identical.
- Before inserting, grep the target file for the first sentence of the
  text about to be inserted; when it is present, skip the slot. A second
  run on the same base changes nothing.
- Register one `SYNC_ROW` per insertion in the synchronization register
  the base deposited.
- After every slot: `grep -rn '{{'` over the delivered paths returns
  nothing; the CI file lints against the target instance; a clean-context
  read of the project skill can state the take-work precondition, the
  draft's first content, the reconciliation, and the finish step.

## Slot table

| Slot | Base location | Insert (asset section) | When |
|---|---|---|---|
| `RELATED_WORK_LINES` | MR template, `## Related work`, after the reference-syntax comment | the specification block: `Spec:`, `Phase:`, records, `Approval:` in the contract's mode | always |
| `ACCEPTANCE_ITEM` | MR template, the checklist item beginning "The change satisfies" | ", or the scenarios of the linked change record" | always |
| `CHECKLIST_ITEMS` | MR template, between the acceptance item and "The documented local checks pass" | the approval item worded for the mode; the archive item worded for the archive mode | always |
| `INTAKE_LINK_FIELD` | task template, one section before `## Acceptance criteria` | `## Specification` with its comment | always |
| `ACCEPTANCE_SOURCE` | task template, the comment under `## Acceptance criteria` | the change-record alternative, "never both" | always |
| `COMPLETION_SOURCE` | the goal's milestone or epic description | link the specifications whose scenarios define it | always |
| `TAKE_WORK_PRECONDITION` | project skill, `## Take and execute work` step 1 | combined: record to the draft first, then stop; split: unmerged specification MR is escalated | always |
| `DRAFT_FIRST_CONTENT` | project skill, `## Take and execute work` step 3 | the record as first push, `Phase: specification`, reserved sections, stop; the wait and the reconciliation for the mode | combined |
| `CREATE_WORK_RULE` | project skill, a `## Create work` section before `## Publish gate` | items carry no acceptance criteria and link the record; task-derived items optional; a non-closing reference for a specification MR | always |
| `FINISH_STEP` | project skill, `## Take and execute work` step 6 | `Phase: implementation`, the reserved lines replaced, the spec-side step for the archive mode, the validator inside the check command | always |
| archive job | `.gitlab-ci.yml` fragment from `assets/gitlab/ci-spec-archive.yml` | the resource-grouped job calling the project's copy of this builder's archive script | automated archiving with OpenSpec |
| local check command | the command the MR checklist's "documented local checks" item and the pipeline already run | the tool's strict validator as one more step; no pipeline edit | the tool ships a validator |
| `KNOWLEDGE_SECTION` | `.agents/knowledge/gitlab-workflow.md`, appended `## Specifications` | the contract's location, the slots filled because of it, the validator's place, the update trigger "when the spec directory or tool changes, re-check every template link" — never the contract's tables | always |
| `SYNC_ROW` | the synchronization register | one row per insertion above | per insertion |
| `MAINTAINER_ACTION` | the platform-settings knowledge, one row | the push path below, with its readback | automated archiving |

Do not add a specification label or work-item type: a spec is a document
in the repository, and its lifecycle lives in the tool's layout. The issue
and incident templates are unchanged.

## Ready-state rules

State them in the knowledge section: a draft may hold the reserved lines
and `Phase: specification`; removing the draft flag needs `Phase:
implementation`, a `Spec:` line, no reserved line in Changes or
Validation, and every checklist item ticked. A pipeline job that checks
the description against these rules is a project asset, not this
builder's.

## Push path

The archive job pushes to the protected default branch. Record the path as
the `MAINTAINER_ACTION` row, and state in the contract that in-request
archiving is in force until it exists: a project access token with write
permission, stored as a masked, protected CI variable, **and** an
allowed-to-push entry for that token's user on the protected branch — a
valid token without the entry is refused. A push made with the token runs
the default-branch pipeline again, including this job, whose second run
finds nothing completed and exits 0; say so in the pipeline knowledge.

Verify the CLI's non-interactive archive flag from its help before
shipping the asset; the asset quotes none.

## Platform-native option

When the contract records committed specification documents with no tool,
the same slots apply: `## Specification` holds a repository path, the
checklist carries the archive item, and no validator joins the check
command. Do not promote work items or the Wiki into the specification
store — a closed item reads as "done", not as a requirement.
