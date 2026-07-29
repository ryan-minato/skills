# REST fallback: reading without glab or MCP

This tier applies when step 3 of "Choose your path" selected it: glab is
absent (or unauthenticated for the target host), no MCP server covers the
work, and the project is public — or a token is set in
`GITLAB_TOKEN`/`GITLAB_ACCESS_TOKEN`. It serves the **read portions of
any operation** — checking templates before a draft, reading an issue or
MR, inspecting a failed pipeline, listing releases. Everything here is
read-only; publishing is impossible on this tier by design. When the task
reaches a write, stop: keep the reviewed draft locally and tell the user
which write was blocked and what tooling it needs (see
[tooling-setup.md](tooling-setup.md)).

## Coverage

The tier covers: issues and their notes, labels, MRs (details, diff,
commits, notes, approvals, pipelines), pipelines/jobs/failure digests,
instance and group search, releases, and tags. It has **no subcommands
for wikis, milestones, boards, iterations, or epics** — on this tier
those reads stop exactly like writes: tell the user what is missing.

All reads go through one bundled script,
[scripts/rest_read.py](scripts/rest_read.py) (Python 3.9+ stdlib). It
talks to the GitLab REST API v4 directly, picks the token up from the
environment automatically (sent as the `PRIVATE-TOKEN` header; values
never printed), and projects responses down to compact fields (`--raw`
for the full payload). `--host` (or `GITLAB_HOST`/`GL_HOST`) selects the
instance — it defaults to gitlab.com only when nothing else is set. When
the glab CLI is installed, the script prints a stderr hint to prefer
glab — heed it: this tier is the last resort.

## Task → invocation

`--project G/P` and `--host HOST` follow the subcommand.

| Task | Invocation |
|---|---|
| List/filter issues | `python3 scripts/rest_read.py issues --project G/P --host HOST [--state closed] [--labels bug] [--search "TEXT"]` |
| Read issue + comments | `python3 scripts/rest_read.py issue --project G/P --host HOST --number N --comments` |
| List labels | `python3 scripts/rest_read.py labels --project G/P --host HOST` |
| List/filter MRs | `python3 scripts/rest_read.py mrs --project G/P --host HOST [--state merged]` |
| Read MR | `python3 scripts/rest_read.py mr --project G/P --host HOST --number N` |
| MR diff / commits / notes / approvals / pipelines | same, plus exactly one of `--diff` `--commits` `--notes` `--approvals` `--pipelines` |
| List pipelines | `python3 scripts/rest_read.py pipelines --project G/P --host HOST [--status failed]` |
| Pipeline jobs | `python3 scripts/rest_read.py jobs --project G/P --host HOST --pipeline-id ID` |
| Failed-job log tails | `python3 scripts/rest_read.py pipeline-failures --project G/P --host HOST --pipeline-id ID [--tail 50]` |
| Search | `python3 scripts/rest_read.py search --scope issues --query "TEXT" --host HOST [--group GROUP]` |
| List releases | `python3 scripts/rest_read.py releases --project G/P --host HOST [--limit 20]` |
| Read one release | `python3 scripts/rest_read.py releases --project G/P --host HOST --tag TAG` (or `--latest`) |
| List tags | `python3 scripts/rest_read.py tags --project G/P --host HOST` |

Exit codes: 0 = read succeeded, 1 = network/HTTP failure (a 429 reports
the `Retry-After` delay instead of sleeping), 2 = bad arguments.

## What a token changes

- Private and internal projects become readable.
- Rate limits rise substantially (unauthenticated gitlab.com is heavily
  limited; the script surfaces `Retry-After` on 429 — stop, don't retry).
- Issue/MR **notes**, **labels**, the **search** API, and **job traces**
  become readable — gitlab.com now auth-gates these even on public
  projects (the script reports each as a clean 401 diagnostic, or per-job
  `log_error` data in `pipeline-failures`).

## Tokenless limits

Public projects only, and only part of their surface: issue and MR
lists/details, diffs, commits, approvals, pipelines, and job listings
work; comments, labels, search, and traces need a token on gitlab.com (a
self-managed instance may be more permissive — its admin decides). Low
rate limits throughout.

## Stay frugal

Unauthenticated quota disappears quickly and the host decides the limit:

- Prefer one targeted read (`issue --number N --comments`) over broad
  listing; never enumerate everything "to be safe".
- Plan the reads before running any: one issue + one search is two
  requests, not ten.
- When the script reports a 429 with its `Retry-After` delay, stop and
  tell the user — never retry into the limit.
