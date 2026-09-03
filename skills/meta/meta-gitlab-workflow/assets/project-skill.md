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

1. Confirm the work item is open. {{SPEC_RULE — under a specification
   contract, one or two sentences: confirm the linked specification is
   approved and treat its scenarios as the acceptance criteria; when
   creating planned work, derive it from the change record's task list and
   link the specification instead of copying acceptance criteria. Delete
   this placeholder otherwise.}} If another person is assigned, stop and
   ask whether duplicate work is intended.
2. Otherwise assign the acting identity, announce the start, and record the
   start time using the project's verified mechanism.
3. Create and push a compliant branch, then open a draft MR immediately. Set
   yourself as assignee and apply the approved labels and milestone.
4. Keep the MR's what/why, changes, links, discoveries, and checklist current.
5. On pause, material change, completion, or abandonment, log time and status.
   On abandonment, explain the handoff and clear the assignee.
6. When acceptance criteria and required checks pass, follow
   {{AUTHORITY_POLICY_PATH — e.g. .agents/knowledge/agent-authority.md}}: a
   green pipeline is evidence, not acceptance. {{READY_POLICY — default:
   stop at the draft and hand the human a decision-ready report (goal
   addressed, tests and pipeline state, actual scope, known risks,
   remaining limitations; their options: request fixes, reject, or accept
   and admit to review) — marking ready and requesting review are the
   human's call. Replace with the granted procedure only if the policy
   delegates it.}} Never edit the policy, protections, or approval rules
   to unblock yourself — propose the change to a human instead.

## Publish gate

GitLab publication cannot be reliably undone: descriptions, comments, commit
messages, tags, attachments, and their notification copies survive deletion.
Every remote or publishable write passes this gate.

1. Assemble the exact final payload as files in a scratch directory outside the
   repository. For a merge request include `title.txt`, `body.md`,
   `commits.txt` from `git log TARGET..SOURCE --format=full`, `diff.patch` from
   `git diff TARGET...SOURCE`, and every attachment — an MR publishes its
   commit messages and diff, not only its description.
2. Review that directory independently: dispatch a clean-context subagent whose
   whole prompt is the review instruction, or, without subagent support, re-read
   every file from disk and note `Review mode: file-only (not clean-context)`.
   Judge only what the files contain.
3. The review checks every line for credentials and secrets, real personal
   data, internal-only hosts/URLs/identifiers, unintended quick actions (any
   line beginning with `/`), unrelated or generated content in the diff, and
   regret-worthy wording. It ends with exactly `SAFE TO PUBLISH: YES` or
   `SAFE TO PUBLISH: NO`; any secret, personal-data, or internal-context
   finding means NO.
4. Treat anything other than a verbatim `SAFE TO PUBLISH: YES` as NO. Fix every
   finding, rebuild the directory, and review again.
5. Confirm applicable user approval, execute non-interactively, and read the
   result back. Published content must be byte-identical to the reviewed
   content; any edit after the verdict requires a fresh review.

## Finish

Report the object/MR links, final state, time recorded, checks, unresolved work,
and any manual or approval-gated action. Update this skill in the same MR when
its commands, templates, labels, paths, settings, or lifecycle policy change.

<!-- Add project-specific routed references and scripts only for selected
planning, pipeline, release, wiki, guardrail, deployment, or MLOps branches. -->
