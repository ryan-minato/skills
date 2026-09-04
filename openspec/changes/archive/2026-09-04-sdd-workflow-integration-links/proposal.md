## Why

Pull requests and issues named a change record by its repository path, which a reviewer has to look up by hand. The lifecycle this branch introduces makes the record the thing under review, so every tracker object that names one should be a link that opens it in one click.

## What Changes

- The pull request template's `Spec:` line and the Task form's Specification field carry a link to the change directory on its branch; `scripts/check_pr_policy.py` accepts the link and bare-URL forms (and still the bare path, for older pull requests).
- `spec-workflow.md`, the `change-workflow` project skill, and `github-checks.md` state the link rule.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository write the `Spec:` line and the Specification field as links to the change directory on the branch, never as bare paths.

## Impact

`.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/task.yml`, `scripts/check_pr_policy.py`, `.agents/knowledge/spec-workflow.md`, `.agents/knowledge/github-checks.md`, `.agents/skills/change-workflow/SKILL.md`.

## Non-goals

The public builders' template assets and the SDD skill's template guidance still say "a `Spec:` line naming the change record"; making them say "linking" is a follow-up skill change.

## Tracked work

No issue: follow-up to `sdd-workflow-integration` requested in conversation.
