---
name: gitlab-project-workflow
description: >-
  Runs this project's GitLab work-item and merge-request lifecycle. Use when
  taking, creating, updating, pausing, or completing a task, issue, incident,
  milestone, or merge request; when diagnosing its pipeline; or when performing
  an approved release, wiki, planning, or repository-setting operation. Not for
  changing this project's lifecycle policy without user approval.
---

# GitLab Project Workflow

Follow the project's recorded host, paths, templates, labels, planning rules,
permissions, and approval boundaries. Read the project entrypoint and the exact
knowledge file routed for the operation before acting.

## Choose the operation path

1. Resolve the GitLab host and full project path from the project record and
   remote. Never assume GitLab.com.
2. Use the project's recorded authenticated tool path. Fall back only as its
   documented capability matrix allows; read-only fallback never writes.
3. Inspect the current object, templates, labels, milestone, assignees,
   permissions, and linked work before drafting a change.

## Take and execute work

1. Confirm the work item is open. If another person is assigned, stop and ask
   whether duplicate work is intended.
2. Otherwise assign the acting identity, announce the start, and record the
   start time using the project's verified mechanism.
3. Create and push a compliant branch, then open a draft MR immediately. Set
   yourself as assignee and apply the approved labels and milestone.
4. Keep the MR's what/why, changes, links, discoveries, and checklist current.
5. On pause, material change, completion, or abandonment, log time and status.
   On abandonment, explain the handoff and clear the assignee.
6. Mark the MR ready only after acceptance criteria and required checks pass.

## Publish gate

Draft the exact final title, body, comment, metadata, attachment, branch, tag,
or setting locally. Review that payload for credentials, confidential
information, sensitive personal data, internal identifiers or URLs, unrelated
content, and unintended quick actions. Continue only with
`SAFE TO PUBLISH: YES` for the unchanged payload and applicable user approval.
Execute non-interactively and read the result back. Any edit invalidates the
verdict.

## Finish

Report the object/MR links, final state, time recorded, checks, unresolved work,
and any manual or approval-gated action. Update this skill in the same MR when
its commands, templates, labels, paths, settings, or lifecycle policy change.

<!-- Add project-specific routed references and scripts only for selected
planning, pipeline, release, wiki, guardrail, deployment, or MLOps branches. -->
