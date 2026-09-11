# Durable GitLab Harness

Read on every approved build. This is the deposit contract that makes the
builder removable.

## Entrypoint

Use the target's existing agent entrypoint. Otherwise create `AGENTS.md` as a
compact map containing project purpose, always-on safety and validation rules,
the GitLab workflow skill location, and exact when-to-read pointers to each
knowledge file. Do not turn it into the full lifecycle manual. Rework
`assets/agents-md-gitlab.md` as the GitLab section of that entrypoint.

## Default structure

When no coherent project convention exists, use:

- `.agents/skills/gitlab-project-workflow/` for recurring task, work-item, MR,
  release, wiki, pipeline, and approved platform-operation procedures;
- `.agents/knowledge/gitlab-workflow.md` — the workflow file the workflow
  builder deposited (objects in use and not used, decomposition, triage,
  planning view), to which this builder appends label semantics, board
  conventions, and the grooming owner; created here in the same shape when
  no builder ran;
- `.agents/knowledge/gitlab/` for platform settings, CI/local-command
  mapping, security/ownership, deployments/releases, Wiki, and optional
  MLOps facts;
- `.gitlab/issue_templates/` and `.gitlab/merge_request_templates/` for human
  and agent intake;
- committed configuration and public policy files at their GitLab- and
  project-conventional locations.

If the framework cannot load project skills, place the recurring procedure in
an existing workflow document reachable from AGENTS.md. Do not generate an
undiscoverable skill.

## Project skill contract

Rework `assets/project-skill.md`. Its description triggers recurring GitLab
work inside this project, not harness construction. Keep the common ownership,
convention discovery, exact-payload safety gate, assignment state machine, and
early draft-MR flow inline. Put optional planning, release, wiki, guardrail,
pipeline, and MLOps branches in skill-local references only when selected.

Copy deterministic scripts only when their branch is selected and record
their runtime. Remove every unused script/reference. The generated skill and
all assets must omit the disposable marker.

## Remote settings as durable knowledge

Committed files cannot prove remote settings. Record the intended protected
refs, approvals, merge method, squash behavior, pipeline requirements,
environments, variables by name (never value), scanners, integrations, boards,
and labels in project knowledge. For each setting state the owner, verification
command or UI path, last verification evidence, and implementation↔harness
update trigger. Rework `assets/platform-settings.md` as that record.

## Synchronization and entropy

Assign one owner for each relationship:

- local validation command ↔ CI job;
- CI job name ↔ merge gate;
- directory/module ↔ CODEOWNERS and area label;
- work-item template quick action ↔ label/type taxonomy;
- every filled extension slot ↔ the paradigm contract it was filled from
  (the filling builder registers the row);
- release tag/changelog ↔ package/deployment automation;
- experiment metadata ↔ training/evaluation implementation;
- public contribution/security statements ↔ internal workflow.

Long-lived, high-change projects add a periodic audit for stale paths,
commands, settings, labels, templates, links, ownership, runner availability,
and unjustified harness thickness.

## Extension slots

The base is paradigm-neutral: nothing in the templates, project skill, or
knowledge presupposes a development paradigm. A paradigm builder — one
whose description claims the contract the entrypoint points to — fills
these slots after this builder has delivered. A slot is a structural
location (a heading, a step, a section), never a surviving placeholder or
an anchor comment, so delivered files read clean and `grep -rn '{{'`
stays empty.

| Slot | Location in the delivered base | What a fill inserts |
|---|---|---|
| `RELATED_WORK_LINES` | MR template, under `## Related work`, after the reference-syntax comment | lines linking the paradigm's record and its phase |
| `ACCEPTANCE_ITEM` | MR template, the checklist item beginning "The change satisfies" | an alternative acceptance source, appended to the item |
| `CHECKLIST_ITEMS` | MR template, between the acceptance item and "The documented local checks pass" | further checklist items; the sensitivity-review item is never edited |
| `INTAKE_LINK_FIELD` | task template, one section immediately before `## Acceptance criteria` | one optional section linking the paradigm's artifact |
| `ACCEPTANCE_SOURCE` | task template, the comment under `## Acceptance criteria` | the alternative source of acceptance, appended |
| `COMPLETION_SOURCE` | the goal's milestone or epic description, its completion statement (shaped by [planning-and-labels.md](planning-and-labels.md)) | what completion links instead of restating |
| `TAKE_WORK_PRECONDITION` | project skill, `## Take and execute work` step 1, after "Confirm the work item is open." | what must exist before the item is taken |
| `DRAFT_FIRST_CONTENT` | project skill, `## Take and execute work` step 3, after "apply the approved labels and milestone." | what the draft's first push carries and what the agent then waits for |
| `CREATE_WORK_RULE` | project skill, one `## Create work` section inserted immediately before `## Publish gate` | how work items derive from the paradigm's artifacts |
| `FINISH_STEP` | project skill, `## Take and execute work` step 6, after "required checks pass," | the paradigm's step before the authority policy applies |
| `KNOWLEDGE_SECTION` | `.agents/knowledge/gitlab-workflow.md`, appended as one `## <Paradigm>` section | the contract's location, the slots filled because of it, the update trigger |
| `SYNC_ROW` | the synchronization register this build deposited | one row per filled slot ↔ its contract |
| `MAINTAINER_ACTION` | the platform-settings knowledge, one row | a setting the paradigm's automation needs, recorded as a maintainer action with its readback |

Fill contract, for the paradigm builder: locate each slot by its structure,
never by a marker; insert, never reword base text; before inserting, grep
for the sentence about to be inserted and skip the slot when it is already
present, so a second run changes nothing; keep the sensitivity-review
checklist item byte-identical; a paradigm's own local check joins the
command the pipeline already runs rather than a new job; register one
`SYNC_ROW` per insertion; then rerun this builder's step-5 checks —
placeholders, links, CI syntax, and a clean-context readback of the
project skill.

## Disposal test

Before the closing step asks the user about deletion, simulate removal by searching target files for
this skill's name, path, disposable marker, and conversation-only references.
Verify every remaining link and procedure. Cleanup is a fresh, explicit user
action after the exact disposable set is shown; building the harness is not
deletion consent.
