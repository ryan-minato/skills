# ops — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

This catalog holds **platform operations** skills: one consolidated,
directly installable playbook per collaboration platform — `github-ops`
and `gitlab-ops` — covering day-to-day forge work (issues, pull/merge
requests, CI, planning structures, releases, wikis, read-only research)
plus the tooling setup that work depends on (CLI and MCP install,
authentication, verification). Conventions and community-file *authoring*
(templates, label taxonomies, commit/release policy, community health
files) lives in the `engineering` catalog's `github-community` /
`gitlab-community` skills.

## Requirements

- **Design floor: a ~30B-parameter local model.** Every operation gets
  exactly one recommended path, selected once per session by the skill's
  "Choose your path" procedure, then decision tables mapping task → exact
  CLI command → MCP capability (or `—` when none exists). No unlabeled
  alternatives, no "you could also". Add a `Done when:` line where
  completion is ambiguous or a weak model may stop early or run past done;
  omit it for steps that are simple and cannot fail.
- **Tool priority is fixed per platform.** `github-ops`: a connected MCP
  server first, then authenticated gh. `gitlab-ops`: authenticated glab
  first, then the GitLab Duo MCP server for rows that name a capability.
  Both platforms end with the same two tiers: the read-only REST fallback
  (public target or a token in the environment; reads only, minimal
  requests), then stop and set up tooling via the skill's own
  `references/tooling-setup.md`. One path per task; `—` rows are the one
  sanctioned switch to the CLI on the MCP path.
- **The REST fallback tier reads only.** It serves the read portions of
  any operation when no MCP server or CLI is available; every write on
  this tier stops with the reviewed draft kept locally and the user told
  which write was blocked and what tooling it needs. Anonymous access is
  tightly rate-limited, so the tier's guidance and its `rest_read.py`
  script both press for the minimum number of requests, and the script
  prints a stderr hint to prefer the CLI whenever one is installed.
- **MCP tools are described by capability, never by name.** MCP tool names
  churn across server versions, and MCP servers self-describe each tool's
  purpose — the agent matches by purpose at runtime. Skill bodies, tables,
  and references write "the MCP tool that reads an issue", never a concrete
  tool name or a `mcp__...` prefix pattern. A GitLab capability cell may
  carry a minimum GitLab version in parentheses. Exact names are allowed
  only for CLI commands, REST/GraphQL endpoints, and bundled scripts.
- **Host-agnosticism is load-bearing for gitlab-ops.** GitLab is routinely
  self-managed: never hardcode `gitlab.com`. Commands, endpoints, config
  examples, and scripts are written against a `HOST` derived from the
  project's git remote or `GITLAB_HOST`/`GL_HOST`; cross-host targeting
  uses `--hostname HOST` (`glab api`/`glab auth`) or `GITLAB_HOST=HOST`.
- **Tier and version gating is marked inline (gitlab-ops).** Premium/
  Ultimate features carry a tier badge; the Duo MCP server requires
  GitLab ≥ 18.6 with Duo enabled and is Beta; an older instance
  legitimately lacks newer capabilities — absence means "use the glab
  column", not an error. GitLab returns **404, not 403**, for features
  above the instance's tier — say so wherever a user could hit it.
  Domains with no MCP coverage (planning, wiki, releases, search) say so
  where they are routed and ship single-path tables.
- **Convention discovery is mandatory before any create.** Each skill
  embeds "Match the project's conventions" with a discovery table
  (templates, labels, milestones, tag schemes, CONTRIBUTING, project
  convention skills / AGENTS.md). Inventing structure parallel to what
  the project already defines is a defect, not a style choice.
- **References are split by branching condition, not topic.** One
  reference file = one load condition; the pointer in SKILL.md (or in the
  parent reference for narrower sub-branches) states that condition
  verbatim — never "see references/ for details". First-level branches are
  what the task operates on; content evaluated on every pass through a
  branch stays inline in that branch's file; the same condition reached
  from several entry points gets one file, not one per door. SKILL.md
  stays around 250 lines: shared blocks, condition-worded routing, and
  cross-domain gotchas only — every domain operation table lives in
  `references/`.
- **Multi-line content is always sent via files** — gh `--body-file`, MCP
  body parameters filled from files, or glab `--description "$(cat
  BODY.md)"` / `-m "$(cat COMMENT.md)"` / `glab api -F "field=@FILE"`
  (glab has no `--body-file`). Never retype reviewed content inline.
- **Non-interactive rule (gitlab-ops).** glab prompts and opens editors by
  default: always pass `--yes` on create and merge commands that support
  it, always supply `-t`/`-d`, and never use `--fill` — it publishes
  generated content that never went through the review gate.
- **Publishing safety.** Publishing = anything that becomes visible to
  others: titles and bodies, comments, labels, milestone/board/project
  names, release notes and assets, commit messages, diffs, attachments,
  branch and tag names. On private or internal projects the content
  becomes visible to every member just as instantly. Each skill embeds
  the canonical pre-publish gate; commit-backed or bulky surfaces (PRs,
  MRs, wiki git pushes) get a self-contained publish-review reference.
  The review procedure is never delegated to a separate skill that might
  not be loaded.
- **Cross-skill canonical sync.** The "Authoring defaults" block and the
  four numbered steps of the pre-publish gate are word-for-word identical
  between `github-ops` and `gitlab-ops`; when editing one copy, update the
  other in the same commit (`grep -rl "Authoring defaults" skills/ops/`
  finds both). Platform-specific gate framing (quick-action hazard,
  no-draft-releases) lives in each skill's own surface sentences.
- Deterministic multi-step logic (ID resolution, version bumping, log
  digestion, REST reads) lives in `scripts/`: python3 ≥3.9 stdlib only,
  invoked with plain `python3`, non-interactive, exit codes 0/1/2, data to
  stdout, diagnostics to stderr, idempotent. `rest_read.py` is the only
  script that opens sockets itself; it never prints token values.
- Sibling skills (including the `engineering` community skills) are named
  with the install-pointer pattern, never path-linked (self-containment).
- Exact CLI subcommand and flag names — and REST/GraphQL endpoint shapes —
  must be re-verified before publishing a skill revision, against
  <https://cli.github.com/manual/> and <https://docs.github.com/en/rest>
  for github-ops, and <https://docs.gitlab.com/cli/> and
  <https://docs.gitlab.com/api/rest/> for gitlab-ops (glab's command
  surface churns; `glab work-items` is experimental; the Epics REST API is
  deprecated). MCP capabilities are deliberately name-free and carry no
  such duty. Each skill's directory is the complete grep scope for its
  platform's command references.

## Disambiguation

Any GitHub operation, research read, or gh/MCP setup → `github-ops` · any
GitLab operation, research read, wiki work, or glab/MCP setup →
`gitlab-ops` · authoring templates, label taxonomies, commit/release
policy, CI validation, or community health files → the `engineering`
catalog's `github-community` / `gitlab-community`.

Boundary rule: the ops skills *operate within* a project's structures
(file issues against templates, apply labels); the community skills
*author* those structures. Reads with no write intent and reads that
immediately precede a write both belong to the same ops skill — routing
happens inside it.

## References

GitHub:

- gh manual: <https://cli.github.com/manual/>
- REST API: <https://docs.github.com/en/rest>
- Projects v2 (concepts + GraphQL): <https://docs.github.com/en/issues/planning-and-tracking-with-projects>
- Milestones REST API: <https://docs.github.com/en/rest/issues/milestones>
- Releases: <https://docs.github.com/en/repositories/releasing-projects-on-github>
- Automatically generated release notes (`.github/release.yml`): <https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes>
- Discussions GraphQL guide: <https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions>
- github-mcp-server: <https://github.com/github/github-mcp-server>

GitLab:

- glab manual: <https://docs.gitlab.com/cli/>
- glab source and releases: <https://gitlab.com/gitlab-org/cli>
- GitLab Duo MCP server: <https://docs.gitlab.com/user/gitlab_duo/model_context_protocol/mcp_server/>
- REST API v4: <https://docs.gitlab.com/api/rest/>
- Releases API: <https://docs.gitlab.com/api/releases/> · Tags API: <https://docs.gitlab.com/api/tags/>
- Changelogs: <https://docs.gitlab.com/user/project/changelogs/>
- Search API: <https://docs.gitlab.com/api/search/>
- Quick actions: <https://docs.gitlab.com/user/project/quick_actions/>
- Labels: <https://docs.gitlab.com/user/project/labels/>
- Wikis API: <https://docs.gitlab.com/api/wikis/> · Boards API: <https://docs.gitlab.com/api/boards/> · Epics API: <https://docs.gitlab.com/api/epics/>
