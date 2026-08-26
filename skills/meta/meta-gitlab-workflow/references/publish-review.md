# Pre-publish Review Procedure

Read before the first remote or publishable write of the session, and deposit
this procedure into the durable project skill. It produces the
`SAFE TO PUBLISH:` verdict that SKILL.md step 4 requires; without it the gate
is a slogan.

GitLab publication is effectively irreversible. Descriptions, comments, commit
messages, branches, tags, attachments, wiki pages, and the notification and
webhook copies they generate cannot be reliably erased from every history and
inbox. Prevention at this gate is the only real control.

## 1. Assemble the exact payload in a scratch directory

Write the payload to files outside the repository; never review from memory or
from an editor buffer. Include, for the surface being published:

| Surface | Files to assemble |
|---|---|
| Work item (task, issue, incident) or comment | `title.txt`, `body.md`, every attachment |
| Merge request | `title.txt`, `body.md`, `commits.txt` from `git log TARGET..SOURCE --format=full`, `diff.patch` from `git diff TARGET...SOURCE`, attachments |
| Wiki page (API or git push) | page titles and bodies, changed wiki files, attachments; for a git push also `commits.txt` and `diff.patch` |
| Release or tag | tag name, tag message, release title, release notes, attached asset list |
| Repository setting or label batch | the exact proposed delta as JSON or a command list |

An MR publishes its commit messages and its whole diff, not only its
description. Reviewing the description alone does not cover the MR.

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
Review the files in <DIR> before they are published on GitLab. Everything in
them becomes visible to everyone who can view the project — on a public
project, the entire internet — and cannot be reliably deleted afterwards. For
a merge request or a wiki git push this includes every commit message and the
whole diff. You have no other context; judge only what the files contain.

1. List the files. If the directory is empty, a file the surface requires
   (for a merge request: `commits.txt` or `diff.patch`) is missing, or any
   file is unreadable, end with SAFE TO PUBLISH: NO.
2. Check every file, line by line, for:
   - Secrets and credentials: tokens, API keys, private keys, passwords,
     connection strings, signing material, cookies, session IDs.
   - Personal data: real personal names, emails, phone numbers, addresses,
     account identifiers, screenshots or attachments exposing people.
     Placeholders such as name@example.com are acceptable.
   - Internal-only context: hostnames, internal URLs, ticket-system links,
     project codenames, unreleased plans, private branch names.
   - Quick actions: any line beginning with `/` (such as /close, /assign,
     /label) executes as a command with the publisher's permissions — flag
     any that are not clearly intended.
   - Unintended content: files in the diff unrelated to the stated change,
     lockfile or editor-config churn, generated output, attachments, or
     screenshots that should not be published.
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
- Record the verdict line verbatim in the session summary, and delete the
  scratch directory once the write completes.

Done when: every publishable payload of the session carries its own verbatim
`SAFE TO PUBLISH: YES`, and the durable project skill states the same
assembly, review, and verdict rules for future agents.
