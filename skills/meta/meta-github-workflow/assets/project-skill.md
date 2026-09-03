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
type, field value, state) is an explicit call — GitHub has no slash
commands.

## Take work

1. Read the issue; confirm it is open and acceptance criteria are
   executable. {{SPEC_RULE — under a specification contract, one sentence:
   confirm the linked specification is approved and treat its scenarios as
   the acceptance criteria. Delete this placeholder otherwise.}} If another
   identity is assigned, stop and ask.
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
labels and `--type` explicitly in the same call (delete the type on a
personal account). A field value cannot be set that way at all: it needs
`gh api repos/{{OWNER_REPO}}/issues/<number>/issue-field-values`, so
priority is a second, separate call. {{SPEC_ISSUES — under a specification
contract, one or two sentences: issues for planned work derive from the
change record's task list, each linking the specification and the
scenarios it closes; never copy acceptance criteria into the issue. Delete
this placeholder otherwise.}} Human-authored
tracking issues and milestones go through the consensus process in
{{KNOWLEDGE_PATH}}/planning.md — never create them unilaterally.

## Publish gate

GitHub publication cannot be reliably undone: bodies, comments, commit
messages, tags, and their notification copies survive deletion, and public
content is indexed within minutes. Every remote or publishable write passes
this gate.

1. Assemble the exact final payload as files in a scratch directory outside
   the repository. For a pull request include `title.txt`, `body.md`,
   `commits.txt` from `git log BASE..HEAD --format=full`, `diff.patch` from
   `git diff BASE...HEAD`, and every attachment — a PR publishes its commit
   messages and diff, not only its description.
2. Review that directory independently: dispatch a clean-context subagent
   whose whole prompt is the review instruction, or, without subagent
   support, re-read every file from disk and note
   `Review mode: file-only (not clean-context)`. Judge only what the files
   contain, never what you remember intending to publish.
3. The review checks every line for credentials and secrets, real personal
   data, internal-only hosts/URLs/identifiers, unrelated or generated content
   in the diff, @-mentions and cross-references that would notify uninvolved
   people, and regret-worthy wording. It ends with exactly
   `SAFE TO PUBLISH: YES` or `SAFE TO PUBLISH: NO`; any secret, personal-data,
   or internal-context finding means NO.
4. Treat anything other than a verbatim `SAFE TO PUBLISH: YES` as NO. Fix
   every finding, rebuild the directory, and review again. A secret that
   reaches GitHub is compromised even after deletion — rotate it.
5. Confirm applicable user approval, execute non-interactively, and read the
   result back. Published content must be byte-identical to the reviewed
   content; any edit after the verdict requires a fresh review.

## Finish

1. Run {{LOCAL_CHECK_COMMAND}}; all checks green
   ({{KNOWLEDGE_PATH}}/checks.md maps jobs to commands — diagnose with
   {{DIGEST_COMMAND_IF_COPIED}}; never fetch full logs, never weaken a
   check).
2. Complete the PR checklist and update the final description. Then follow
   {{AUTHORITY_POLICY_PATH — e.g. .agents/knowledge/agent-authority.md}}:
   green checks are evidence, not acceptance. {{READY_POLICY — default:
   stop at the draft and hand the human a decision-ready report (goal
   addressed, tests and CI state, actual scope, known risks, remaining
   limitations; their options: request fixes, reject, or accept and admit
   to review) — gh pr ready and requesting review are the human's call.
   Replace with the granted procedure only if the policy delegates it.}}
   {{AUTO_MERGE_POLICY — whether gh pr merge --auto is approved here}}.
   Never edit the policy, protections, or required checks to unblock
   yourself — propose the change to a human instead.
3. Merge closes the linked issue via the closing keyword; verify it
   closed. {{RELEASE_POINTER — when and how releases are cut, per
   knowledge}}.

<!-- Add project-specific routed references and copied scripts only for
selected planning, Projects, release, guardrail, Actions-diagnosis, or ML
branches. -->
