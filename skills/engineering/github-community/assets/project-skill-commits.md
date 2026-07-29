---
name: {{REPO_NAME}}-commits
description: >
  Writes commit messages for {{REPO_NAME}} following this repository's
  Conventional Commits rules, and validates them with the committed
  checker before committing. Use when committing in this repository —
  "commit this", "write the commit message", "what type should this
  commit use", or when a commit-message check fails.
---

# {{REPO_NAME}} commits

Write and validate commit messages for this repository. The enforced
rules live in `scripts/check_commits.py` (CONFIG block) and are
documented in {{CONVENTION_DOC_PATH}}.

## The format

```
type(scope)!: subject
```

- Types: {{TYPES_LIST}}
- Scope: {{SCOPE_RULE}}
- Subject: imperative mood, whole title at most {{SUBJECT_MAX}}
  characters, no trailing period, English.
- Add a body (blank line after the title, wrapped at
  {{BODY_LINE_MAX}} characters) when the why is not obvious from the
  diff; body explains why, not how.
- Breaking changes: `!` after the type/scope plus a `BREAKING CHANGE:`
  footer.

## Choosing the type

Pick the type from the **dominant intent** of the change:
{{TYPES_TABLE}}

When a commit seems to need two types, it is usually two commits —
split it rather than picking the vaguer type.

## Validate before committing

Write the message to a file and run the repository's own validator —
never commit an unvalidated message:

```bash
python3 scripts/check_commits.py --file COMMIT_MSG.txt
```

Fix every finding and re-run until it prints `OK`, then commit with
`git commit -F COMMIT_MSG.txt`. The same validator runs in CI over the
whole PR range, so a message that fails locally will fail the PR.

Done when: the validator prints OK for the exact message used in the
commit.
