# Shaping the GitHub Base for the Contract

Read in step 7 when the evidenced platform is GitHub and the platform
builder has delivered its base. The base registers its extension slots in
its own durable-harness reference (`## Extension slots`): a slot is a
heading, a step, or a field id, never a marker. This file says what to
insert into each slot for the contract's shape, approval mode, and archive
mode; the exact texts live in `assets/github/`. Never
re-decide a contract fact here.

## Fill contract

- Locate each slot by its structure — the heading, the step number, the
  field id the base's table names — never by searching for a placeholder.
- Insert; never reword base text. The security checklist item stays
  byte-identical: the checklist workflow keys on the word "secrets", so
  keep that word out of every inserted item.
- Before inserting, grep the target file for the first sentence of the
  text about to be inserted; when it is present, skip the slot. A second
  run on the same base changes nothing.
- Register one `SYNC_ROW` per insertion in the synchronization register
  the base deposited, pairing the inserted text with the contract fact it
  came from.
- After every slot: `grep -rn '{{'` over the delivered paths returns
  nothing; every workflow parses; a body built from the shaped template
  passes the checklist workflow; a clean-context read of the project skill
  can state the take-work precondition, the draft's first content, the
  reconciliation, and the finish step.

## Slot table

| Slot | Base location | Insert (asset section) | When |
|---|---|---|---|
| `RELATED_WORK_LINES` | PR template, `## Related work`, after the closing-keyword comment | the specification block: `Spec:`, `Phase:`, records, `Approval:` in the contract's mode | always |
| `ACCEPTANCE_ITEM` | PR template, the checklist item beginning "Acceptance criteria" | ", or the scenarios of the linked change record" | always |
| `CHECKLIST_ITEMS` | PR template, between the acceptance item and the security item | the approval item worded for the mode; the archive item worded for the archive mode | always |
| `INTAKE_LINK_FIELD` | task and feature forms, before the field with id `acceptance` | the optional `spec` input | always |
| `ACCEPTANCE_SOURCE` | task and feature forms, the `acceptance` field's `description` | the change-record alternative, "never both" | always |
| `COMPLETION_SOURCE` | tracking-issue body, `## Observable completion` | link the specifications whose scenarios define the goal | always |
| `TAKE_WORK_PRECONDITION` | project skill, `## Take work` step 1 | combined: record to the draft first, then stop; split: unmerged specification PR is escalated | always |
| `DRAFT_FIRST_CONTENT` | project skill, `## Take work` step 3 | the record as first push, `Phase: specification`, reserved sections, stop; the wait and the reconciliation for the mode | combined |
| `CREATE_WORK_RULE` | project skill, `## Create issues` | issues carry no acceptance criteria and link the record; task-derived issues optional; `Refs #N` for a specification PR | always |
| `FINISH_STEP` | project skill, `## Finish` step 2 | `Phase: implementation`, the reserved lines replaced, the spec-side step for the archive mode, the validator inside the check command | always |
| archive workflow | `.github/workflows/spec-archive.yml` from `assets/github/workflow-spec-archive.yml` | the serialized job calling the project's copy of this builder's archive script | automated archiving with OpenSpec |
| local check command | the command the PR template's first checklist item and the checks workflow already run (`justfile`, `Makefile`, or a script) | the tool's strict validator as one more step; no workflow edit | the tool ships a validator |
| `KNOWLEDGE_SECTION` | `.agents/knowledge/github-workflow.md`, appended `## Specifications` | the contract's location, the slots filled because of it, the validator's place, the update trigger "when the spec directory or tool changes, re-check every template link" — never the contract's tables | always |
| `SYNC_ROW` | the synchronization register | one row per insertion above | per insertion |
| `MAINTAINER_ACTION` | `platform-settings.md`, one row | the push path below, with its readback | automated archiving |

Do not add a "Specification" issue type or label: a spec is a document in
the repository, and its lifecycle lives in the tool's layout, not in issue
metadata. The bug report form is unchanged: a bug's acceptance baseline is
the expected behavior, which under a spec-anchored contract is the spec's
current requirement.

## Ready-state rules

State them in the knowledge section and, when the project keeps a body
check, give that check these rules: a draft may hold the reserved lines
and `Phase: specification`; a ready pull request needs `Phase:
implementation`, a `Spec:` line, no reserved line in Changes or
Validation, and every checklist item ticked. The base's checklist workflow
enforces headings and the security item only; a body check that knows the
phase is a project asset, not this builder's.

## Push path by owner type

The archive workflow pushes to the protected default branch. Record the
path as the `MAINTAINER_ACTION` row, and state in the contract that
in-request archiving is in force until it exists:

- **Organization-owned repository.** The GitHub Actions app is added as a
  bypass actor on the default branch's ruleset; the job pushes with
  `GITHUB_TOKEN`. Such pushes trigger no further workflows, so the job runs
  the strict validator itself (the asset does).
- **User-owned repository.** The rulesets API refuses the Actions app as a
  bypass actor ("must be part of the ruleset source or owner
  organization"). The push needs either a deploy key with write access
  (checkout with the key; the private key is a repository secret) or a
  GitHub App installed on the repository whose installation token the job
  mints. Pushes made with either identity trigger workflows — including
  the archive workflow itself, whose second run finds nothing completed and
  exits 0 — so name that in `checks.md`.

Verify the CLI's non-interactive archive flag from its help before
shipping the asset; the asset quotes none.

## Platform-native option

When the contract records committed specification documents with no tool,
the same slots apply: the `spec` input holds a repository path, the
checklist carries the archive item, no validator joins the check command,
and `checks.md` says the checklist is the only gate. Do not promote issues
or Discussions into the specification store — a closed issue reads as
"done", not as a requirement.
