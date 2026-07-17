---
name: git-commit
description: >
  Executes the full git commit workflow as ordered safety gates before anything is
  committed. Use when creating a git commit or saving work to version control —
  "commit this", "checkpoint my work", "save this to git"; when finished work sits
  uncommitted at the end of a task; or when staged changes need splitting into atomic
  commits. Not for drafting a message in a specific convention without committing, for
  standalone secret/PII scans, or for authoring a repository's commit rules.
compatibility: >
  scripts/validate_message.py requires uv and Python 3.11+ (stdlib only).
---

# Git Commit

Work through the gates in order; each one either passes or names the action to
take. Skipping a gate is a decision the user makes, not the agent.

## Rule precedence

Resolve every convention dimension (message format, type/emoji lists, scope,
footers, identity, check commands) at the highest level that states it; lower
levels only fill what remains open.

1. **Explicit documentation** — `AGENTS.md`, then `CONTRIBUTING.md`, then
   `README.md`; follow pointers between them. Explicit statements are final.
2. **Tool configuration** — commit-message linters (`commitlint.config.*`,
   `.commitlintrc*`), `.gitmessage`, hook configs.
3. **Git history** — `git log --oneline -20` for casing, scope, and footer habits.
4. **Defaults in this skill.**

Track which level resolved each dimension: when everything came from level 3–4,
suggest (after the commit) documenting the convention in `AGENTS.md` or
`CONTRIBUTING.md` so future contributors and agents stop guessing.

## Gate 1 — Atomicity

Run `git status` and `git diff --cached --stat` (stage the intended files first if
nothing is staged). One commit = one logical change. The gate fails when the
staged diff mixes unrelated concerns: independent bug fixes, refactoring plus
behavior change, feature code plus unrelated cleanups.

On failure, propose a grouping with a one-line rationale per group and confirm
with the user, then read
[references/splitting-and-recovery.md](references/splitting-and-recovery.md) for
the selective-staging split procedure. Proceed as one commit only when the user
asked for a combined commit.

## Gate 2 — Sensitivity

Review `git diff --cached` (added lines) for:

- Private keys (`-----BEGIN ... PRIVATE KEY-----`) and certificate bundles.
- Credential assignments: values for keys named like `api_key`, `secret`,
  `token`, `password`, `credential`.
- Provider key shapes, e.g. `AKIA` + 16 uppercase alphanumerics (AWS).
- Personal data: real email addresses (placeholders like `name@example.com` are
  fine), names, phone numbers, or other PII the project did not already publish.

On any finding, stop and show it to the user; commit only what they explicitly
clear. This skill pairs with `sensitivity-check` for a deeper scan. If it is not
installed, install it from https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill sensitivity-check

Then check the committer identity: `git config user.email`. If the project's
documentation states an email rule, apply it. Otherwise, when the address is a
personal-provider one (gmail.com, outlook.com, icloud.com, qq.com, …), warn the
user that public history will expose it and suggest a platform noreply address —
proceed only on their confirmation.

## Gate 3 — Hooks and checks

Detect the hook manager and make sure it is installed before committing:

| Present | Manager | If not installed |
|---|---|---|
| `.pre-commit-config.yaml` | pre-commit | `pre-commit install --install-hooks` |
| `.husky/` | husky | verify `.husky/pre-commit` is executable |
| `lefthook.yml` | lefthook | `lefthook install` |
| `.git/hooks/pre-commit` | raw hook | verify it is executable |

Run the project's own check entrypoint if one exists (`just check`,
`make check`, package-manager script, or the commands the CI config runs).
A failing check or hook stops the workflow: read the output, fix the cause, and
rerun. Avoid `--no-verify` unless the user asks for it. If a hook rewrites files
(formatters often do), read the recovery section of
[references/splitting-and-recovery.md](references/splitting-and-recovery.md).

## Gate 4 — Message

Format comes from the resolved conventions. When the project uses Conventional
Commits or gitmoji, this skill pairs with `conventional-commits` / `gitmoji` for
drafting rules. If not installed, install from
https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill conventional-commits
    npx skills add ryan-minato/skills --skill gitmoji

In every format, the same structural rules apply:

- Title ≤ 50 characters, imperative mood, no trailing period; title-only is the
  default, add a body when the why is not obvious from the diff.
- Blank line between title and body; body lines ≤ 72 characters; the body
  explains why, not how.
- Footers (issue refs, `BREAKING CHANGE:`) each on their own line after a blank
  line.

Validate the draft with
[`scripts/validate_message.py`](scripts/validate_message.py) and fix every
reported error:

```bash
uv run scripts/validate_message.py --file /path/to/COMMIT_EDITMSG
uv run scripts/validate_message.py --message "fix: reject expired tokens"
```

## Commit

`git commit -m "title"` for title-only messages; write the message to a file and
use `git commit -F <file>` when there is a body (quoting multi-line strings in
shells corrupts them easily). Confirm the result with `git log -1 --stat`, and
deliver the post-commit convention suggestion if Rule precedence flagged it.

## Gotchas

- An empty `git diff --cached` usually means nothing is staged, not that there is
  no work — check `git status` before concluding.
- Hooks that modify files leave staged and working copies out of sync; re-add the
  files and commit again rather than committing the stale staging.
- A commit-msg hook rejecting the message means the draft, not the hook, is wrong
  in the project's eyes — fix the draft to the configured rules.
- The 50/72 limits count characters (codepoints), not bytes — CJK text and emoji
  do not triple the count the way byte-counting suggests.
