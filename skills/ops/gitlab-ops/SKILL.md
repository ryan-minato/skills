---
name: gitlab-ops
description: >
  GitLab operations on gitlab.com or any self-managed host — issues, merge requests,
  pipelines, planning (milestones, iterations, boards, epics, labels), releases,
  wikis, project research, and glab/MCP tooling setup. Use when filing, commenting
  on, or closing an issue; when opening, reviewing, approving, or merging an MR, or
  checking why its pipeline is red; when managing milestones, boards, epics, or
  label lifecycle; when cutting a release, tagging, or generating a changelog; when
  reading or writing wiki pages ("document this in the wiki"); when investigating
  any GitLab project, even without write access; when a GitLab issue (#N), MR (!N),
  pipeline, or wiki URL is the material at hand; or when glab or the GitLab Duo MCP
  server is missing or unauthenticated ("set up glab"). Authoring description
  templates, label taxonomies, commit/release policy, or CONTRIBUTING belongs to
  gitlab-community; GitHub work to github-ops.
license: Apache-2.0
compatibility: >
  Bundled scripts require Python 3.9+ (stdlib only). rest_read.py needs
  outbound HTTPS to the GitLab host; pipeline_log_digest.py needs an
  authenticated glab CLI; next_version.py reads local git tags.
---

# GitLab Operations

One skill for day-to-day GitLab work on gitlab.com or any self-managed
host: issues, merge requests, pipelines, planning structures, releases,
wikis, read-only research, and setting up the tooling all of it uses.
Authoring a project's description templates, label taxonomy,
commit/release policy, or community files belongs to `gitlab-community`;
GitHub work to `github-ops`. If either is needed and not installed,
install it from https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill gitlab-community

## Choose your path (do this first, once per session)

1. Run `glab auth status`. If it exits 0 and lists the target host, use
   the **glab** column of every table. For a self-managed host, check that
   host specifically: `glab auth status --hostname HOST`.
2. Otherwise, look at the tools available in this session. If a connected
   MCP server provides GitLab tools for the work at hand (each tool's
   description states its purpose; names vary across server versions), use
   the **MCP** column, picking the tool whose description matches the
   row's capability — but only for rows that name one. Rows marked `—`
   have no MCP tool, whole domains (planning, wiki, releases, search
   endpoints) have no MCP coverage at all, and an older self-managed
   instance may lack a capability entirely: for those tasks, glab is
   required.
3. Otherwise, for the **read portions of the task only** — if the target
   project is public, or a token is set in
   `GITLAB_TOKEN`/`GITLAB_ACCESS_TOKEN` even though glab is missing — use
   the read-only REST fallback: read
   [references/rest-fallback.md](references/rest-fallback.md) and run
   [scripts/rest_read.py](scripts/rest_read.py) with `--host HOST`. Reads
   only, and unauthenticated access is rate-limited by the host (a 429
   reports `Retry-After` — stop, never retry into the limit) — make the
   fewest requests the task allows. For any write on this tier: stop,
   keep the reviewed draft locally, and tell the user which write was
   blocked and what tooling it needs.
4. Otherwise stop and tell the user GitLab tooling is not set up, then
   offer to set it up now: read
   [references/tooling-setup.md](references/tooling-setup.md) and work
   through it.
5. Use one path for the whole task; `—` rows and no-MCP domains are the
   one sanctioned switch to glab on the MCP path. Never mix glab, MCP,
   and REST within a single operation.

## Identify the host and project

Run `git remote get-url origin`. The host is the part right after
`https://` or the `@` (GitLab is often self-managed — never assume
`gitlab.com`). The project path is everything after the host, with any
trailing `.git` stripped; GitLab paths can nest (`group/subgroup/project`
is one project — keep the full path). The group path is the project path
minus its last segment; group-level structures (group milestones,
iterations, epics, group boards, group labels, group wikis) live there.
If there is no origin remote, or the user named a different project, use
that instead — research often targets a project other than the current
checkout. Substitute the full path wherever the tables show `G/P` (glab:
`-R G/P`; MCP: the project identifier parameter; URL-encode `/` as `%2F`
inside `glab api` endpoint paths, or use `:fullpath`/`:group` inside a
checkout). Inside the project's checkout, glab resolves the host from the
remote on its own; outside it, pass `--hostname HOST` to
`glab api`/`glab auth` and set `GITLAB_HOST=HOST` for other command
groups.

## Route by task

Read the file for the branch the task is on — now, before acting — and
only that file:

| When the task | Read |
|---|---|
| Operates on issues (file, read, comment, triage, close, list, search) | [references/issues.md](references/issues.md) |
| Operates on merge requests (open, comment, ready, approve, merge) | [references/merge-requests.md](references/merge-requests.md) |
| Needs a pipeline or job read, or its failure digested — from an MR or directly | [references/pipelines.md](references/pipelines.md) |
| Manages milestones, labels, iterations, boards, or epics (glab only — no MCP tools) | [references/planning.md](references/planning.md) |
| Cuts, edits, reads, or deletes releases or tags (glab only) | [references/releases.md](references/releases.md) |
| Reads or writes wiki pages (glab only) | [references/wiki.md](references/wiki.md) |
| Searches the instance, a group, or a project | [references/search.md](references/search.md) |
| Is blocked on missing or broken glab/MCP tooling, or asks to set it up | [references/tooling-setup.md](references/tooling-setup.md) |

Read-only research on a project you are not acting in uses the same
files' read rows and needs no publish gate. The branch that owns a write
also owns the reads immediately preceding it — stay in its file rather
than switching.

## Sending content (non-interactive rules)

- Send multi-line text through a file, never an inline shell string:
  write the body to a file first, then pass it with `-d "$(cat BODY.md)"`
  or `-m "$(cat COMMENT.md)"` (glab has no `--body-file`; command
  substitution does not re-expand file contents). Where a command lacks a
  text flag, `glab api -F "field=@FILE"` is the fallback.
- glab prompts and opens editors by default: always pass `-y`/`--yes` on
  create and merge commands that support it, and always supply `-t`/`-d`
  so no editor opens.
- Never use `--fill` — it publishes generated content that never went
  through the review gate.

## Match the project's conventions (before any create)

Before creating anything, discover what the project already defines and
use it — never invent parallel structure. Check the rows relevant to what
is being created:

| Artifact | How to check |
|---|---|
| Issue description templates (issue create) | List `.gitlab/issue_templates/` (locally, or `glab api "projects/:fullpath/repository/tree?path=.gitlab/issue_templates"`); if one matches the report type, use its body as the scaffold and fill every section — templates may deliberately embed quick actions, keep them |
| MR templates (MR create) | `.gitlab/merge_request_templates/` (locally, or the same tree endpoint); `Default.md` auto-applies in the web UI — the drafted description must follow it section by section |
| Contributing rules (MR create) | `CONTRIBUTING.md` (root or `docs/`) — MR titling, target-branch, and review rules stated there are binding |
| Merge settings (MR create) | `glab api projects/:fullpath -F json` fields `merge_method` and `squash_option` — they decide whether `--squash` is wanted, forbidden, or implied |
| Labels | `glab label list -R G/P -F json` — apply only labels that already exist (an unknown `-l NAME` silently creates a label) |
| Open milestones | `glab milestone list -R G/P` — assign only existing milestones |
| Tag scheme and changelog config (release create) | `git tag --sort=-v:refname \| head -20`; `.gitlab/changelog_config.yml` present and commits carry `Changelog:` trailers means generated notes are the project's default |

If a project-level convention skill or an AGENTS.md conventions section
covers this task, follow it over this skill's defaults.
Done when: each relevant artifact was checked and the draft uses the
project's existing structures (or the user approved new ones).

## Authoring defaults

Write all published text — titles, bodies, comments, notes — as
professional, concise prose. Default to English unless the user or the
project's own conventions call for another language. State facts and
requests directly; no filler, and no emojis unless the project's existing
content uses them. The project's templates and conventions win over these
defaults.

## Pre-publish gate (mandatory)

Everything you send becomes visible the moment the call succeeds — to the
whole internet on public projects, and to every member just as instantly on
private or internal ones: title, body, every comment, labels, commit
messages, the full diff, attachment contents, and the branch name. A line
starting with `/` in any body or comment can execute as a GitLab quick
action (for example `/close`). Each branch file's intro states its
surface-specific exposure facts. Before ANY call that creates or edits
such content:

1. Prefer a clean-context subagent review when one is available and the
   surface is not trivial. Give it only the exact final text or files under
   review, with no extra intent or reassurance.
2. Otherwise review the exact final text yourself. For short text fully
   visible in context, inspect it directly. For attachments, screenshots,
   generated bodies, long notes, or content too large to inspect reliably
   inline, write the exact outgoing content to a scratch directory and
   review those files from disk.
3. Check every artifact for secrets or credentials, personal data, internal
   identifiers or URLs, unintended quick actions, accidental unrelated
   content, and wording a maintainer would regret publishing. Any finding
   means `SAFE TO PUBLISH: NO`.
4. Publish only after the verdict is exactly `SAFE TO PUBLISH: YES`. On
   `NO`, fix every finding and review the exact final content again. Never
   edit-and-publish without re-review.

Never publish unreviewed content. Only the user may skip this gate,
explicitly; record the skip in your summary.
Done when: a `SAFE TO PUBLISH: YES` verdict exists for the exact content
being sent.

The gate applies to any call that creates or edits public text or
metadata. Pure reads and lists carry no new content and skip it.
Commit-backed surfaces run the file-based procedure named in their branch
file: [references/publish-review-mr.md](references/publish-review-mr.md)
for MR publishing, [references/publish-review-wiki.md](references/publish-review-wiki.md)
for wiki git pushes.

## Gotchas (cross-domain)

- If no available MCP tool's description matches a row's capability, that
  capability is missing from the connected server (older instances have
  partial tool lists) — use the glab column instead of guessing.
- GitLab returns **404, not 403**, for features above the instance's tier
  or license — on a 404 for a Premium/Ultimate row, report the tier
  requirement instead of retrying. Probe the version with
  `glab api version` when a feature might be too new for the host.
- Issue and MR iids are per-project and independent: `#42` and `!42` are
  different objects, and a "not found" on one kind can mean the number
  belongs to the other.
- A body or comment line starting with `/` executes as a quick action
  with your permissions — never let one through unintended (the gate
  checks), but deliberately used they are the supported way to do things
  glab lacks flags for.
- Output flags are inconsistent across glab commands (`-O` vs `-F` for
  JSON) — check `--help` before assuming; glab's command surface churns
  across releases, so a missing subcommand means update glab
  (`glab check-update`) rather than hunting for alternate spellings.
- The REST tier can only read, and unauthenticated it is rate-limited by
  the host; the script surfaces `Retry-After` on 429 — stop and tell the
  user rather than retrying.
