# Publish review prompt

Load from step 6 of SKILL.md. Dispatch a clean-context subagent whose
entire prompt is the block below with `<DIR>` replaced by the scratch
directory; add nothing else, because explaining what you meant destroys
the independence that makes the review worth running. Without subagent
support, apply the same checklist yourself, reading every file from disk,
and write `Review mode: file-only (not clean-context)` above the verdict.

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
     Placeholders such as name@example.com and GitHub noreply addresses are
     acceptable.
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

Treat any last line other than exactly `SAFE TO PUBLISH: YES` as NO.
Published content must be byte-identical to the reviewed content. Record
the verdict line verbatim in the handoff and delete the scratch directory
once the write completes.
