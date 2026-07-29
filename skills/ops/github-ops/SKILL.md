---
name: github-ops
description: >
  GitHub operations — issues, pull requests, CI checks and Actions, Discussions,
  planning (milestones, labels, Projects), releases, read-only repo research, and
  gh/MCP tooling setup. Use when filing, commenting on, triaging, or closing an issue
  ("file an issue", "comment on #N"); when opening, reviewing, marking ready, or
  merging a PR, or checking its CI ("did the checks pass", "why is CI red"); when
  managing milestones, labels, or project boards; when cutting a release, tagging a
  version, or drafting release notes; when investigating any repository, even
  without write access — upstream issues, discussions, failed runs, releases; when a
  GitHub issue, PR, run, or release URL is the material at hand; or when gh or the
  GitHub MCP server is missing,
  unauthenticated, or failing ("set up GitHub MCP", "gh auth failed"). Authoring
  templates, label taxonomies, commit/release policy, or community health files
  belongs to github-community; GitLab work to gitlab-ops.
license: Apache-2.0
compatibility: >
  Bundled scripts require Python 3.9+ (stdlib only). rest_read.py needs
  outbound HTTPS to api.github.com and github.com; run_log_digest.py and
  project_fields.py need an authenticated gh CLI; next_version.py reads
  local git tags.
---

# GitHub Operations

One skill for day-to-day GitHub work: issues, pull requests, CI runs and
checks, Discussions, planning structures, releases, read-only research on
any repository, and setting up the tooling all of it uses. Authoring a
repository's templates, label taxonomy, commit/release policy, or
community health files belongs to `github-community`; GitLab work to
`gitlab-ops`. If either is needed and not installed, install it from
https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill github-community

## Choose your path (do this first, once per session)

1. Look at the tools available in this session. If a connected MCP server
   provides GitHub tools for the work at hand (each tool's description
   states its purpose; names vary across server versions), use the **MCP**
   column of every table, picking the tool whose description matches the
   row's capability.
   The MCP column applies only to rows that name a capability; rows marked
   `—` have no MCP tool — those rows use the gh column instead: check
   `gh auth status` before running one, and if gh is not authenticated,
   stop and tell the user that row needs gh.
2. Otherwise run `gh auth status`. If it exits 0, use the **gh** column.
3. Otherwise, for the **read portions of the task only** — if the target
   repository is public, or a token is set in `GH_TOKEN`/`GITHUB_TOKEN`
   even though gh is missing — use the read-only REST fallback: read
   [references/rest-fallback.md](references/rest-fallback.md) and run
   [scripts/rest_read.py](scripts/rest_read.py). Reads only, and anonymous
   access is limited to 60 requests/hour (search: 10/minute) — make the
   fewest requests the task allows. For any write on this tier: stop, keep
   the reviewed draft locally, and tell the user which write was blocked
   and what tooling it needs.
4. Otherwise stop and tell the user GitHub tooling is not set up, then
   offer to set it up now: read
   [references/tooling-setup.md](references/tooling-setup.md) and work
   through it.
5. Use one path for the whole task; rows marked `—` are the one sanctioned
   switch to gh on the MCP path. Never mix MCP, gh, and REST within a
   single operation.

## Identify the repository

Run `git remote get-url origin`. The owner/repo pair is the path right
after the host, with any trailing `.git` stripped. If there is no origin
remote, or the user named a different repository, use that instead.
Substitute the pair wherever the tables show `O/R` (gh: `-R O/R`; MCP: the
owner and repo parameters). Research often targets a repository other than
the current checkout: when the user names one, that name is `O/R` and the
git remote is only the default.

## Route by task

Read the file for the branch the task is on — now, before acting — and
only that file:

| When the task | Read |
|---|---|
| Operates on issues (file, read, comment, triage, close, list, search) | [references/issues.md](references/issues.md) |
| Operates on pull requests (open, comment, ready, review, merge) | [references/pull-requests.md](references/pull-requests.md) |
| Needs a workflow run, check, or its logs read — from a PR's red checks or directly | [references/ci-runs.md](references/ci-runs.md) |
| Touches Discussions | [references/discussions.md](references/discussions.md) |
| Manages milestones, label lifecycle, or Projects boards | [references/planning.md](references/planning.md) |
| Cuts, edits, reads, or deletes releases or tags | [references/releases.md](references/releases.md) |
| Is blocked on missing or broken gh/MCP tooling, or asks to set it up | [references/tooling-setup.md](references/tooling-setup.md) |

Read-only research on a repository you are not acting in uses the same
files' read rows and needs no publish gate. The branch that owns a write
also owns the reads immediately preceding it — stay in its file rather
than switching.

## Match the project's conventions (before any create)

Before creating anything, discover what the repository already defines and
use it — never invent parallel structure. Check the rows relevant to what
is being created:

| Artifact | How to check |
|---|---|
| Issue templates and forms (issue create) | List `.github/ISSUE_TEMPLATE/` (locally, or `gh api repos/O/R/contents/.github/ISSUE_TEMPLATE`); a bare `.github/ISSUE_TEMPLATE.md` also counts. If any exist, read [references/use-issue-forms.md](references/use-issue-forms.md) and draft the body against the matching template |
| PR template(s) (PR create) | `.github/PULL_REQUEST_TEMPLATE.md` (also `PULL_REQUEST_TEMPLATE.md` at root or in `docs/`), or multiple under `.github/PULL_REQUEST_TEMPLATE/` — if one exists, the PR body follows its structure section by section |
| Contributing rules (PR create) | `CONTRIBUTING.md` (root, `.github/`, or `docs/`) — PR titling, base-branch, and review rules stated there are binding |
| Labels | `gh label list -R O/R`, or the MCP tool that lists repository labels — apply only labels that already exist |
| Open milestones | `gh api repos/O/R/milestones -q '.[].title'` — assign only existing milestones |
| Tag scheme and notes config (release create) | `git tag --sort=-v:refname \| head -20` for prefix and semver shape; `.github/release.yml` present means generated notes are the project's default |

If a project-level convention skill or an AGENTS.md conventions section
covers this task, follow it over this skill's defaults.
Done when: each relevant artifact was checked and the draft uses the
repository's existing structures (or the user approved new ones).

## Authoring defaults

Write all published text — titles, bodies, comments, notes — as
professional, concise prose. Default to English unless the user or the
project's own conventions call for another language. State facts and
requests directly; no filler, and no emojis unless the project's existing
content uses them. The project's templates and conventions win over these
defaults.

## Pre-publish gate (mandatory)

Everything you send becomes public the moment the call succeeds: title, body,
every comment, labels, commit messages, the full diff, attachment contents,
and the branch name. Each branch file's intro states its surface-specific
exposure facts. Before ANY call that creates or edits public content:

1. Prefer a clean-context subagent review when one is available and the
   surface is not trivial. Give it only the exact final text or files under
   review, with no extra intent or reassurance.
2. Otherwise review the exact final text yourself. For short text fully
   visible in context, inspect it directly. For attachments, screenshots,
   generated bodies, long notes, or content too large to inspect reliably
   inline, write the exact outgoing content to a scratch directory and
   review those files from disk.
3. Check every artifact for secrets or credentials, personal data, internal
   identifiers or URLs, accidental unrelated content, and wording a
   maintainer would regret publishing. Any finding means
   `SAFE TO PUBLISH: NO`.
4. Publish only after the verdict is exactly `SAFE TO PUBLISH: YES`. On
   `NO`, fix every finding and review the exact final content again. Never
   edit-and-publish without re-review.

Never publish unreviewed content. Only the user may skip this gate,
explicitly; record the skip in your summary.
Done when: a `SAFE TO PUBLISH: YES` verdict exists for the exact content
being sent.

The gate applies to any call that creates or edits public text or metadata.
Pure reads and lists carry no new content and skip it. Read
[references/publish-review.md](references/publish-review.md) when the gate
covers a commit-backed or bulky surface (a PR, long release notes,
assets) — it holds the file-based review procedure.

## Gotchas (cross-domain)

- If no available MCP tool's description matches a row's capability, that
  capability is missing from the connected server — use the gh column for
  that row instead of guessing.
- The GitHub MCP server groups tools into optional toolsets; the
  Discussions, Actions, and labels groups are not in its default set. If
  those capabilities are missing while other GitHub tools exist, the
  toolset must be enabled server-side — covered in
  [references/tooling-setup.md](references/tooling-setup.md).
- Issues and PRs share one number space: a bare `#N` can be either, and a
  "not found" error for one kind can mean the number belongs to the other.
- Send multi-line bodies through a file, never an inline shell string:
  `--body-file FILE` (gh) or the MCP body parameter filled from the file.
- gh `--json` field names are camelCase (`updatedAt`, not `updated_at`).
- Labels and milestones passed at create time must already exist in the
  repository; otherwise the call fails. The conventions section above
  exists so this never surprises you.
- The REST tier can only read, and unauthenticated it is tightly
  rate-limited; the script reports exhaustion with the reset time — stop
  and tell the user rather than retrying.
