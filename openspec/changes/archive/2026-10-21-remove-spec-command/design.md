## Context

`spec-command.yml` runs on `issue_comment` from `main`'s copy with `pull-requests: write`, snapshots the pull request head through the API with `scripts/spec_changes.ts snapshot`, and replies with `spec_changes.ts show` or `status`. It is one of three privileged workflows `github-checks.md` lists and the validator's `workflows` check reads. `show` (with `--url-prefix` and `--max-chars`) exists only for that reply. `snapshot` and `labels` also serve `spec / labels`; `status`, `check`, and `archive` serve the recipes. Binding constraint: nothing under `skills/` changes. See proposal.md for motivation.

## Placement

| Item | File and section | Check that proves it |
|---|---|---|
| Workflow removed | `.github/workflows/spec-command.yml` deleted | `git ls-files .github/workflows`; no `issue_comment` trigger in `.github/` |
| Privileged list | `github-checks.md ## Privileged workflows` line without `spec-command.yml` | `just validate` (the `workflows` and `checks` checks) |
| `show` mode removed | `scripts/spec_changes.ts` command table, parser, `showMarkdown`, `cmdShow`, usage; `spec_changes_test.ts` | `just test`; `spec_changes.ts show` exits 2 as an unknown command |
| Docs | `github-checks.md` table and healthy-run list, `spec-workflow.md ## Request automation`, `agent-authority.md`, `ARCHITECTURE.md`, `change-workflow` §5 | `git grep` for `spec / command`, `spec-command`, `/spec ` outside `skills/`, `openspec/specs/`, archives returns nothing |

## External impact

- `skills/**`, `openspec/specs/**`, archived changes: unchanged; `git diff --stat origin/main -- skills openspec/specs openspec/changes/archive` is empty.
- GitHub: after merge no `spec / command` run starts; #96 and #101 are closed as not planned with a comment.

## Decisions

- **Remove rather than fix**: fixing #96 and #101 would grow a token-holding comment handler for a convenience the local recipes already give. Rejected: keeping the workflow read-only (a reply needs a write token).
- **Drop `show`**: unused modes are dropped, as the harness rebuild did for others; `status` stays because `just spec-changes status` uses it.
- **Keep `issue_comment` privileged in the validator**: with no workflow listed for it, any future comment-triggered workflow fails the check until it is listed deliberately.

## Risks / Trade-offs

- [Someone relied on `/spec` replies] → `just spec-changes status` and the files on the branch give the same information; the maintainer requested the removal.

## Verification plan

Script and tool harnesses:
- `just check` passes; `just test` passes with the `show` tests removed.
- `./scripts/spec_changes.ts show` exits 2 (unknown command); `just spec-changes status` and `just spec-check origin/main HEAD --draft` still work.
- `git grep -nE 'spec / command|spec-command|/spec (show|status|help)'` outside `skills/`, `openspec/specs/`, and archives returns nothing; `git grep -n issue_comment -- .github` returns nothing.
- `git diff --stat origin/main -- skills openspec/specs openspec/changes/archive` is empty.

Skipped:
- A live `/spec` comment after merge: nothing answers it by design.
