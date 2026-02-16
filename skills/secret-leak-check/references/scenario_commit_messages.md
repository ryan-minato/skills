# Scenario: Commit Message Review

Review commit messages for leakage risk.

## Default behavior

If user does not explicitly disable commit message checking:

1. Inspect pending commit messages that are in current explicit scan range.
2. By default, do not infer and scan the full PR commit range.
3. Scan commit subject and body for secrets/privacy leakage.

If user explicitly asks to scan all commits in a PR, scan commit messages for every commit in that PR range.

## What to detect in commit messages

- Direct tokens, API keys, passwords, bearer values
- Connection strings, hostnames with credentials, private endpoints
- Personal email address, phone number, legal/real full name when unnecessary
- Database/table details that should remain internal in open context

## Typical risky patterns

- "temp token: ..."
- "debug account/password"
- "my email is ..."
- "call me at ..."
- "created table x with columns ..." (when confidential)

## Result expectations

For each risky commit message include:

- Commit hash
- Risk excerpt (masked when needed)
- Risk reason
- Suggested remediation (amend/reword/squash/rewrite history)
