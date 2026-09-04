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
- the specification section in work-item and MR templates ↔ the artifact
  paths in the specification contract;
- the change request shape and archive mode in the specification contract
  ↔ the project skill's take-work and finish steps, the MR template's two
  specification items, and the archive job's presence;
- release tag/changelog ↔ package/deployment automation;
- experiment metadata ↔ training/evaluation implementation;
- public contribution/security statements ↔ internal workflow.

Long-lived, high-change projects add a periodic audit for stale paths,
commands, settings, labels, templates, links, ownership, runner availability,
and unjustified harness thickness.

## Disposal test

Before the closing step asks the user about deletion, simulate removal by searching target files for
this skill's name, path, disposable marker, and conversation-only references.
Verify every remaining link and procedure. Cleanup is a fresh, explicit user
action after the exact disposable set is shown; building the harness is not
deletion consent.
