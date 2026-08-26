# ops — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

This catalog holds **platform operations** skills: one consolidated,
directly installable playbook per collaboration platform — currently
`github-ops` — covering day-to-day forge work (issues, pull requests, CI,
planning structures, releases, read-only research) plus the tooling setup
that work depends on (CLI and MCP install, authentication, verification).
Conventions and community-file *authoring* (templates, label taxonomies,
commit/release policy, community health files) lives in the `engineering`
catalog's `github-community` skill. Building a GitLab project's complete
lifecycle harness — including the durable project-level skills that then
run its day-to-day operations — belongs to the disposable `meta` catalog's
`meta-gitlab-workflow`.

## Requirements

- **Design floor: a ~30B-parameter local model.** Every operation gets
  exactly one recommended path, selected once per session by the skill's
  "Choose your path" procedure, then decision tables mapping task → exact
  CLI command → MCP capability (or `—` when none exists). No unlabeled
  alternatives, no "you could also". Add a `Done when:` line where
  completion is ambiguous or a weak model may stop early or run past done;
  omit it for steps that are simple and cannot fail.
- **Tool priority is fixed per platform.** `github-ops`: a connected MCP
  server first, then authenticated gh, then the read-only REST fallback
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
  tool name or a `mcp__...` prefix pattern. Exact names are allowed only
  for CLI commands, REST/GraphQL endpoints, and bundled scripts.
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
- **Multi-line content is always sent via files** — gh `--body-file` or MCP
  body parameters filled from files. Never retype reviewed content inline.
- **Publishing safety.** Publishing = anything that becomes visible to
  others: titles and bodies, comments, labels, milestone/project names,
  release notes and assets, commit messages, diffs, attachments, branch
  and tag names. On private or internal repositories the content becomes
  visible to every member just as instantly. Each skill embeds the
  canonical pre-publish gate; commit-backed or bulky surfaces (PRs,
  release assets) get a self-contained publish-review reference. The
  review procedure is never delegated to a separate skill that might not
  be loaded.
- Deterministic multi-step logic (ID resolution, version bumping, log
  digestion, REST reads) lives in `scripts/`: python3 ≥3.9 stdlib only,
  invoked with plain `python3`, non-interactive, exit codes 0/1/2, data to
  stdout, diagnostics to stderr, idempotent. `rest_read.py` is the only
  script that opens sockets itself; it never prints token values.
- Skills are fully independent of each other: sibling skills (including
  the `engineering` community skills) are named at most for
  disambiguation — never path-linked, never accompanied by install
  instructions, and never depended on behaviorally (self-containment).
- Exact CLI subcommand and flag names — and REST/GraphQL endpoint shapes —
  must be re-verified before publishing a skill revision, against
  <https://cli.github.com/manual/> and <https://docs.github.com/en/rest>.
  MCP capabilities are deliberately name-free and carry no such duty. Each
  skill's directory is the complete grep scope for its platform's command
  references.

## Disambiguation

Any GitHub operation, research read, or gh/MCP setup → `github-ops` ·
authoring templates, label taxonomies, commit/release policy, CI
validation, or community health files → the `engineering` catalog's
`github-community` · building or systematically repairing a GitLab
project's complete lifecycle harness → the `meta` catalog's
`meta-gitlab-workflow`.

Boundary rule: the ops skills *operate within* a project's structures
(file issues against templates, apply labels); the community skills
*author* those structures. Reads with no write intent and reads that
immediately precede a write both belong to the same ops skill — routing
happens inside it.

## References

- gh manual: <https://cli.github.com/manual/>
- REST API: <https://docs.github.com/en/rest>
- Projects v2 (concepts + GraphQL): <https://docs.github.com/en/issues/planning-and-tracking-with-projects>
- Milestones REST API: <https://docs.github.com/en/rest/issues/milestones>
- Releases: <https://docs.github.com/en/repositories/releasing-projects-on-github>
- Automatically generated release notes (`.github/release.yml`): <https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes>
- Discussions GraphQL guide: <https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions>
- github-mcp-server: <https://github.com/github/github-mcp-server>
