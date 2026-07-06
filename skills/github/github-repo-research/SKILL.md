---
name: github-repo-research
description: >
  Read-only research over a GitHub repository's Discussions and Actions
  history: lists and reads discussions and their comments, lists workflow
  runs, inspects jobs, and extracts failed-log excerpts through a digest
  script that keeps huge logs out of context. Use when investigating what
  happened in a repository — "what was decided in that discussion",
  "search the discussions", "show recent workflow runs", "why did the
  nightly build fail", "get the logs from that Actions run", or "list
  failed CI runs".
license: Apache-2.0
compatibility: >
  scripts/run_log_digest.py requires Python 3.9+ (stdlib only) and an
  authenticated gh CLI.
---

# GitHub Repo Research

This skill is read-only: it never posts, edits, or changes anything, so it
has no publish gate. If the task turns into posting — commenting on a
discussion, filing an issue, fixing CI via a PR — switch to
`github-issues` or `github-pull-requests`. If GitHub tooling itself is
missing or misconfigured, that is `github-tooling-setup` work.

## Choose your path (do this first, once per session)

1. Look at the tools available in this session. If any tool name contains
   `issue_read`, `pull_request_read`, or a `github` MCP server prefix (for
   example `mcp__github__...`), the GitHub MCP server is connected: use the
   **MCP** column of every table below.
2. Otherwise run `gh auth status`. If it exits 0, use the **gh** column.
3. Otherwise stop and tell the user GitHub tooling is not set up. This skill
   pairs with `github-tooling-setup`. If it is not installed, install it from
   https://github.com/ryan-minato/skills.git:

       npx skills add ryan-minato/skills --skill github-tooling-setup

4. Use one column for the whole task. Never mix MCP and gh in one operation.

## Identify the repository

Run `git remote get-url origin`. Take the part after `github.com/` or
`github.com:`, strip a trailing `.git`, and split on `/` to get `OWNER` and
`REPO`. If there is no `origin` remote, ask the user for them. Substitute
wherever the tables show `O/R` (gh: `-R O/R`; MCP: the `owner` and `repo`
parameters).

## Discussions

| Task | MCP tool | gh |
|---|---|---|
| List discussions | `list_discussions` (orderBy `updatedAt`) | see [references/discussions-gh.md](references/discussions-gh.md) |
| Read one discussion | `get_discussion` (`discussionNumber`) | see references/discussions-gh.md |
| Read its comments | `get_discussion_comments` | see references/discussions-gh.md |
| List categories | `list_discussion_categories` | see references/discussions-gh.md |

gh has no first-class discussions command (the gh team declined to add
one). On the gh path, read
[references/discussions-gh.md](references/discussions-gh.md) whenever the
task involves Discussions — it contains complete copy-paste GraphQL
queries for every row above, plus pagination and search.

## Actions

| Task | MCP tool | gh |
|---|---|---|
| List workflow runs | `actions_list` method=`list_workflow_runs` | `gh run list -R O/R --limit 20 --json databaseId,displayTitle,workflowName,headBranch,status,conclusion,createdAt` |
| Inspect one run | `actions_get` method=`get_workflow_run` | `gh run view RUN_ID -R O/R` |
| List a run's jobs | `actions_list` method=`list_workflow_jobs` | `gh run view RUN_ID -R O/R --json jobs` |
| Failed-log excerpt | `get_job_logs` (`failed_only: true`, `tail_lines: 100`, `return_content: true`) | [scripts/run_log_digest.py](scripts/run_log_digest.py): `python3 scripts/run_log_digest.py --repo O/R --run-id RUN_ID [--tail 50]` |

NEVER run bare `gh run view --log` or fetch full logs: run logs can be
megabytes and will flood the context. Always request failed-only output
with a tail limit — on the gh path the digest script enforces exactly
that, printing one JSON object with the run's status, its failed jobs,
each job's failed steps, and the last lines of each failed job's log.

Done when (for a "why did it fail" task): the failing job and step are
named and the relevant error lines are quoted.

Read [references/actions-recipes.md](references/actions-recipes.md) when
the task needs artifacts, run timing/usage, or filtered run listings
beyond the table above.

## Gotchas

- MCP tool names have changed across github-mcp-server versions. If a tool
  named in a table is absent, list the github server's available tools and
  pick the same-purpose name; if none matches, fall back to the gh column.
- The `discussions` and `actions` toolsets are not in the GitHub MCP
  server's default set. If those tools are missing while other github
  tools exist, the toolset must be enabled — local server: the
  `GITHUB_TOOLSETS` environment variable; remote server: the per-toolset
  URL such as `https://api.githubcopilot.com/mcp/x/discussions`.
  Configuring that belongs to `github-tooling-setup`.
- `gh run list` shows GitHub Actions workflow runs only; check runs
  reported by external CI apps do not appear there.
- Discussion numbers are per-repository and not shared with issue/PR
  numbers: `discussionNumber` 42 and issue #42 are unrelated items.
