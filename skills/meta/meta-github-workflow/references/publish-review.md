# Pre-publish Review Procedure

Read before the first remote or publishable write of the session, and deposit
this procedure into the durable project skill. It produces the
`SAFE TO PUBLISH:` verdict that SKILL.md step 4 requires; without it the gate
is a slogan.

GitHub publication is effectively irreversible. Issue and pull-request bodies,
comments, review threads, commit messages, branches, tags, releases, and the
notification, email, and webhook copies they generate cannot be reliably erased
— and public content is indexed and mirrored within minutes. Prevention at this
gate is the only real control.

## 1. Assemble the exact payload in a scratch directory

Write the payload to files outside the repository; never review from memory or
from an editor buffer. Include, for the surface being published:

| Surface | Files to assemble |
|---|---|
| Issue or comment | `title.txt`, `body.md`, every attachment |
| Pull request | `title.txt`, `body.md`, `commits.txt` from `git log BASE..HEAD --format=full`, `diff.patch` from `git diff BASE...HEAD`, attachments |
| Review or review comment | the review body and every inline comment, plus the lines they quote |
| Release | tag name, release title, notes (generated notes included, read as published text), attached asset list |
| Discussion or wiki page | title and body, attachments; for a wiki git push also `commits.txt` and `diff.patch` |
| Repository setting or label batch | the exact proposed delta as JSON or a command list |

A pull request publishes its commit messages and its whole diff, not only its
description. Reviewing the description alone does not cover the PR.

## 2. Choose the review mode

| Environment | Mode |
|---|---|
| The agent can dispatch a subagent that starts with its own clean context | **Clean-context review** — preferred |
| No subagent support | **File-only fallback** |

**Clean-context review:** dispatch one subagent whose entire prompt is the
review prompt below with `<DIR>` replaced by the scratch directory path. Add
nothing else — explaining what you meant destroys the independence that makes
the review worth running. The subagent's last output line is the verdict.

**File-only fallback:** re-read every file in the directory from disk, top to
bottom, applying the same checklist. Judge only what the files contain, never
what you remember intending to publish. Add the line
`Review mode: file-only (not clean-context)` directly above the verdict.

## Review prompt

```
Review the files in <DIR> before they are published to GitHub. Everything in
them becomes visible to everyone who can see the repository — on a public
repository, the entire internet, indexed and mirrored — and cannot be
reliably deleted afterwards. For a pull request this includes every commit
message and the whole diff. You have no other context; judge only what the
files contain.

1. List the files. If the directory is empty, a file the surface requires
   (for a pull request: `commits.txt` or `diff.patch`) is missing, or any
   file is unreadable, end with SAFE TO PUBLISH: NO.
2. Check every file, line by line, for:
   - Secrets and credentials: tokens, API keys, private keys, passwords,
     connection strings, signing material, cookies, session IDs.
   - Personal data: real personal names, emails, phone numbers, addresses,
     account identifiers, screenshots or attachments exposing people.
     Placeholders such as name@example.com are acceptable.
   - Internal-only context: hostnames, internal URLs, ticket-system links,
     project codenames, unreleased plans, private branch names.
   - Unintended content: files in the diff unrelated to the stated change,
     lockfile or editor-config churn, generated output, attachments, or
     screenshots that should not be published.
   - Unintended addressing: @-mentions and team mentions that would notify
     people who did not ask to be involved, and issue or PR references that
     would post a cross-link into someone else's thread.
   - Quality: confusing, hostile, speculative, or regret-worthy wording in
     title, body, comments, commit messages, or visible diff text.
3. Report one line per finding: file, short excerpt with secret values
   masked, category, required fix. If there are none, say "No findings."
4. Your last output line must be exactly `SAFE TO PUBLISH: YES` or
   `SAFE TO PUBLISH: NO`. Any finding of secrets, credentials, personal
   data, or internal-only context means NO.
```

## Verdict handling

- Treat any review whose last line is not exactly `SAFE TO PUBLISH: YES` as NO.
- On NO: fix every finding, rebuild the scratch directory from the fixed
  content, and review again. Never publish a partially fixed payload.
- Published content must be byte-identical to the reviewed content. Any edit
  after the verdict — including a typo fix — invalidates it and requires a
  fresh review.
- A secret that reaches GitHub is compromised even after the object is
  deleted: rotate it, do not just remove the text.
- Record the verdict line verbatim in the session summary, and delete the
  scratch directory once the write completes.

Done when: every publishable payload of the session carries its own verbatim
`SAFE TO PUBLISH: YES`, and the durable project skill states the same
assembly, review, and verdict rules for future agents.
