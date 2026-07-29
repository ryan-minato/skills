# Community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY

Loaded when the task adds or audits a project's community files. GitLab
has **no default community health file mechanism**: there is no
`.github`-style account repository serving defaults, no `FUNDING.yml`
rendering, and no community-profile checklist — those are GitHub-only
mechanisms (`github-community` covers them). On GitLab these files are
conventional repository files: ship them in the project root (or `docs/`
if the project keeps prose there) and keep the standard filenames so
humans and tools find them.

What GitLab itself surfaces: a root `CONTRIBUTING.md` is linked from the
project's web UI (verify the exact surfacing on the target instance
against <https://docs.gitlab.com> rather than assuming — it has moved
between the project overview and the MR creation flow across versions).
`CODE_OF_CONDUCT.md` and `SECURITY.md` get no special UI treatment; their
value is the convention itself.

## Per-file guidance

| File | Approach |
|---|---|
| `CONTRIBUTING.md` | This file owns its placement and non-MR sections (project setup, where to ask questions, issue etiquette); the MR-flow section comes from [mr-conventions.md](mr-conventions.md) and the commit rules from [commit-conventions.md](commit-conventions.md) — link, don't duplicate |
| `CODE_OF_CONDUCT.md` | Adopt an established code rather than writing one: fetch the current Contributor Covenant text from https://www.contributor-covenant.org/, fill in the enforcement contact with the user, and keep the attribution notice — the Covenant is CC BY 4.0, so attribution is required when its text is committed |
| `SECURITY.md` | Write it with the user: where to report privately (a security contact address, or the instance's confidential-issue flow — a confidential issue is visible to project members with at least the configured role), supported versions, response expectations. Do not point reporters at the public tracker |
| `SUPPORT.md` / `GOVERNANCE.md` | Plain conventional files with no GitLab-side behavior; write them with the user only when the project genuinely needs them (multiple support channels, multiple maintainers) |

## Gotchas

- Issue and MR *templates* are not community files — they live under
  `.gitlab/` and belong to [issue-conventions.md](issue-conventions.md)
  / [mr-conventions.md](mr-conventions.md).
- Group-level file inheritance does not exist for these files; each
  project carries its own copies. Instance or group *description
  templates* (Premium) cover issue/MR templates only, not community
  files.
- Everything here is a plain repository file: it takes effect when
  merged to the default branch, like any other file.
