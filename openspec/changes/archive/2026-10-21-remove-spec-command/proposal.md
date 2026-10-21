## Why

The `/spec` comment command (`spec / command`) answers `/spec show` and `/spec status` on pull requests with a write token, under `issue_comment`. It duplicates what an agent or the maintainer reads locally with `just spec-changes status` and the files themselves, has two open defects (#96, #101), and keeps a third privileged workflow and a document-rendering mode beside a token. The maintainer judged it not worth its attack surface or its upkeep.

## What Changes

- Remove `.github/workflows/spec-command.yml`, so no workflow in this repository runs on `issue_comment`.
- Remove the `show` mode of `scripts/spec_changes.ts` (with `--url-prefix`, `--max-chars`, and their tests), whose only caller was that workflow; `snapshot`, `status`, `check`, `archive`, and `labels` stay.
- `github-checks.md`: drop the `spec / command` row and its healthy-run line; the privileged-workflow list names only `spec-labels.yml` (`pull_request_target`) and `issue-triage.yml` (`issues`). The `workflows` check keeps `issue_comment` as a privileged trigger, so a new comment-triggered workflow fails until it is listed.
- Remove the `/spec` mentions from `spec-workflow.md`, `agent-authority.md`, `ARCHITECTURE.md`, and the `change-workflow` skill.
- Close #96 and #101 as not planned, now that nothing answers `/spec`.

## Skills touched

repository change.

## Installed behavior

Agents read change status locally (`just spec-changes status`, `just spec-check`) instead of through a pull request comment; nothing in this repository answers `/spec`, and no workflow holds a token on comment events.

## Impact

- Files: `.github/workflows/spec-command.yml` (removed); `scripts/spec_changes.ts`, `scripts/spec_changes_test.ts`; `.agents/knowledge/github-checks.md`, `spec-workflow.md`, `agent-authority.md`, `harness-maintenance.md` where it names the workflow; `ARCHITECTURE.md`; `.agents/skills/change-workflow/SKILL.md`.
- Untouched: everything under `skills/` — the `openspec-workflow` and `spec-kit-workflow` skills keep delivering their `/spec` assets to user projects; `openspec/specs/`; archived changes.
- GitHub: #96 and #101 closed as not planned with a comment when the change is proposed; `spec / command` stops running after merge.

## Non-goals

- No change to the public skills' `/spec` assets or their specs.
- No change to `spec / labels`, the approval label, or `checks / spec`.

## Deliverable and pre-authorized actions

A pull request from `chore/remove-spec-command`. Pre-authorized by the maintainer in conversation: pushes to that branch, the draft pull request and edits to its body, and closing #96 and #101 with a comment.

## Tracked work

#96 and #101, closed as not planned with this proposal.
