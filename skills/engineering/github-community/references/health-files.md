# Community health files

Loaded when the task adds or audits community health files, sets up the
account-wide `.github` default repository, or completes the community
profile. GitHub's supported set: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
discussion category forms, `FUNDING.yml`, `GOVERNANCE.md`, issue and PR
templates plus `ISSUE_TEMPLATE/config.yml`, `SECURITY.md`, and
`SUPPORT.md`.

## Placement and defaults

- Within a repository, GitHub looks for each file in `.github/`, then the
  repository root, then `docs/` — first hit wins. Pick one location and
  keep all health files together (root is the most discoverable;
  `.github/` keeps the root uncluttered — follow what the repo already
  does).
- A public repository named `.github` under the account (org or user)
  provides **default** community health files for every repository of
  that account that lacks its own file of that type. The same
  `.github/` > root > `docs/` order applies inside it. A repo-local file
  always overrides the inherited default — check during assessment
  whether a file is already inherited before adding a local copy.
- **LICENSE cannot be a default file**: license files must live in each
  repository so they are included when the project is cloned, packaged,
  or downloaded.
- The repository's community profile checklist (Insights → Community
  Standards) is the completeness gauge — use it to verify which files
  GitHub recognizes after the changes land on the default branch.

## Per-file guidance

| File | Approach |
|---|---|
| `CONTRIBUTING.md` | This file owns its placement and non-PR sections (project setup, where to ask questions, issue etiquette); the PR-flow section comes from [pr-conventions.md](pr-conventions.md) and the commit rules from [commit-conventions.md](commit-conventions.md) — link, don't duplicate |
| `CODE_OF_CONDUCT.md` | Adopt an established code rather than writing one: fetch the current Contributor Covenant text from https://www.contributor-covenant.org/, fill in the enforcement contact with the user, and keep the attribution notice — the Covenant is CC BY 4.0, so attribution is required when its text is committed |
| `SECURITY.md` | Copy [assets/security-template.md](assets/security-template.md) and settle the placeholders with the user: where to report (prefer GitHub private vulnerability reporting when enabled), supported versions, response expectations |
| `SUPPORT.md` | Copy [assets/support-template.md](assets/support-template.md) and settle the placeholders: where to ask questions (Discussions, chat, tracker) and what belongs in issues instead |
| `GOVERNANCE.md` | Judgment-heavy — no template. Write it with the user: decision-making roles, how maintainers are added, how disputes resolve. Only worth shipping for projects with several maintainers |
| `FUNDING.yml` | Lives at `.github/FUNDING.yml`; keys are platform names with account values, for example: `github: [USER1, USER2]`, `open_collective: PROJECT`, `custom: ["https://example.com/donate"]`. Only add platforms the user actually uses |
| Discussion category forms | `.github/DISCUSSION_TEMPLATE/<category-slug>.yml`, one per discussion category, using the issue-forms element syntax ([issue-forms-schema.md](issue-forms-schema.md)) |
| Issue / PR templates | Owned by [issue-conventions.md](issue-conventions.md) / [pr-conventions.md](pr-conventions.md) — route there |

## Gotchas

- Health files count for the community profile only once they are on the
  default branch, in a recognized location, with the exact expected
  filename.
- An org-level default file is invisible in the repo's own tree —
  contributors see it on GitHub's UI surfaces (new-issue chooser,
  contributing banner) even though `git ls-files` shows nothing; do not
  "fix" its absence with a duplicate copy unless the repo needs to
  diverge from the org default.
- The `.github` default repository must be public to serve defaults;
  private copies serve nothing.
