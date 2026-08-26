---
name: github-project-workflow
description: >-
  Runs this project's GitHub issue and pull-request lifecycle. Use when
  taking, creating, updating, or completing an issue, tracking issue,
  milestone, or pull request; when diagnosing a failing check; or when
  performing an approved release, planning, or repository-setting
  operation. Not for changing this project's lifecycle policy without
  user approval.
---

# {{PROJECT_NAME}} GitHub Workflow

Tool path: {{TOOL_PATH — e.g. "authenticated gh; gh api for milestones
and discussions"}}. Every metadata change (labels, assignees, milestone,
type, state) is an explicit call — GitHub has no slash commands.

## Take work

1. Read the issue; confirm it is open and acceptance criteria are
   executable. If another identity is assigned, stop and ask.
2. Assign yourself, re-read, and confirm you are the sole assignee.
3. `gh issue develop -c {{ISSUE_NUMBER_PLACEHOLDER}}` to create and check
   out the linked branch; push it and open a draft PR immediately with
   `Closes #N` in the body. The draft PR is the claim and the work log.
4. Keep the PR description current; comment major discoveries and
   decisions. {{CONFIDENTIALITY_RULE — what must never appear in issues,
   PRs, or logs}}.
5. Abandon by un-assigning, closing the draft with a status comment, and
   leaving the issue open.

## Create issues

Non-interactive creation ignores templates: build the body by mirroring
the form's `### <label>` headings ({{FORM_PATHS}}), then apply the form's
labels{{AND_TYPE_IF_ORG}} explicitly in the same call. Human-authored
tracking issues and milestones go through the consensus process in
{{KNOWLEDGE_PATH}}/planning.md — never create them unilaterally.

## Publish gate

Draft the exact final payload → review it for credentials, confidential
information, personal data, internal identifiers, and unrelated content →
proceed only with `SAFE TO PUBLISH: YES` → confirm approval covers the
write → execute non-interactively → read back and compare. Any edit after
review restarts the gate.

## Finish

1. Run {{LOCAL_CHECK_COMMAND}}; all checks green
   ({{KNOWLEDGE_PATH}}/checks.md maps jobs to commands — diagnose with
   {{DIGEST_COMMAND_IF_COPIED}}; never fetch full logs, never weaken a
   check).
2. Complete the PR checklist, update the final description, `gh pr ready`.
   {{AUTO_MERGE_POLICY — whether gh pr merge --auto is approved here}}.
3. Merge closes the linked issue via the closing keyword; verify it
   closed. {{RELEASE_POINTER — when and how releases are cut, per
   knowledge}}.

<!-- Add project-specific routed references and copied scripts only for
selected planning, Projects, release, guardrail, Actions-diagnosis, or ML
branches. -->
