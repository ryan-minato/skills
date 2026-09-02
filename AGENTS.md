# AGENTS.md

Agent entrypoint for this repository. Read this before making changes.

## Purpose

This repository is a library of [Agent Skills](https://agentskills.io):
self-contained skill directories that users install into their agents with
`npx skills add ryan-minato/skills`. Public skills live under
`skills/<catalog>/<skill-name>/`. It is long-lived and collaboratively
maintained.

## Catalogs

Public skills are grouped into catalogs under `skills/`. The catalog list,
each catalog's contract, and the procedure for adding one live in the
Catalogs section of `ARCHITECTURE.md`.

## Repository Map

- `skills/<catalog>/<skill-name>/` — public, distributable skills.
- `.agents/skills/` — project-only workflow skills (real directories) plus
  symlinks to every public skill, so this repo's agents can use them.
  `.claude/skills` (Claude Code discovery) and `.github/skills` (Copilot
  code review) are both symlinks to `.agents/skills`.
- `.agents/knowledge/` — git-tracked local knowledge base.
- `.github/` — GitHub Issue intake, labels, pull request policy, and Actions.
- `.claude-plugin/marketplace.json` — plugin marketplace: one plugin per
  non-empty catalog.
- `scripts/` — repository tooling (validators, longer custom logic).
- `ARCHITECTURE.md` — how these mechanisms fit together.

## Core Conventions

- **Language**: all harness files, skill content, code, and comments are
  written in English. The root `README.md` and every catalog `README.md`
  have a content-identical Chinese translation in `README.zh.md` beside them.
- **Skill self-containment**: public skills must not reference anything
  outside their own directory — no links to repo files, no dependencies on
  other skills, and no `README.md` inside a skill root. To build on another
  skill in this repo, instruct installing it:
  `npx skills add ryan-minato/skills` (repo:
  `https://github.com/ryan-minato/skills.git`). Full standards:
  `.agents/knowledge/skill-quality.md`.
- **Checks**: always run checks through justfile recipes (`just check`),
  never ad-hoc equivalents, so results are consistent everywhere.
- **Commits**: non-merge/revert commits use Conventional Commits in English;
  scope is the modified skill name(s), `", "`-separated; omit the scope for
  non-skill or repo-wide changes. Classify changes to distributable skills by their effect: `fix`
  corrects wrong, misleading, overly restrictive, or overly permissive
  installed behavior; `feat` adds a new capability; and `refactor`
  restructures without changing behavior. Correction takes precedence over
  improvement. Use `docs` only for supporting documentation that does not
  change an installed skill; file format does not decide the type. Template:
  `.gitmessage` (installed by `just setup`).
- **Task management**: GitHub is the only task-management platform. Issues are
  optional; the platform-neutral rules live in
  `.agents/knowledge/project-workflow.md`.
- **Agent authority**: `.agents/knowledge/agent-authority.md` is the sole
  source of truth for ready, review, merge, release, and escalation authority.
- **Workflow**: use the `change-workflow` project skill for tracked work. It
  requires atomic commits, explicit authorization for remote writes, and the
  draft-to-ready pull-request lifecycle.

## When To Read What

- Starting any proactive change, or preparing a branch, commit series, or
  PR handoff → use the `change-workflow` project skill.
- Creating, splitting, or closing tracked work; starting a change request; or
  proposing management structure → read
  `.agents/knowledge/project-workflow.md`.
- Marking work ready, requesting review, approving, merging, releasing,
  deploying, or resolving an authority question → read
  `.agents/knowledge/agent-authority.md`.
- Creating or modifying any skill → use the `skill-authoring` project skill;
  it routes to `.agents/knowledge/skill-quality.md`, the catalog's
  `CONTEXT.md`, subagent-capability-gated behavioral tests, and independent
  script and static validation.
- Catalog-specific rules and references → `skills/<catalog>/CONTEXT.md`
  (catalog-scoped material belongs there, not in the global references).
- External documentation URLs → `.agents/knowledge/references.md`.
- Labels, Issue Forms, or PR policy changed → inspect `.github/` and keep every
  referenced label and exact validated heading synchronized.
- GitHub checks or remote settings changed → read
  `.agents/knowledge/github/checks.md` and
  `.agents/knowledge/github/platform-settings.md`.
- Repo mechanics (symlinks, plugin marketplace, sync design) →
  `ARCHITECTURE.md`.

## Validation

- `just check` — everything (validator, lint, pre-commit hooks).
- `just validate` — skill layout and harness consistency only.
- `just check-skill <dir>...` — lint specific skill directories while drafting.
- `just lint` — ruff over `scripts/`.
- `just test` — unit tests for repository tooling.
- `just commit-gate` — pre-commit safety gate over staged changes.
- `just gen-marketplace` — regenerate `marketplace.json` skills[] after
  adding or removing a public skill.

## Keep In Sync

| When this changes | Update |
|---|---|
| Public skill added/removed | Symlink in `.agents/skills/`, catalog `README.md` + `README.zh.md`, and `.claude-plugin/marketplace.json` (`just gen-marketplace`) |
| Catalog added/removed | Catalog scaffold, the Catalogs section in `ARCHITECTURE.md`, `.claude-plugin/marketplace.json` (a plugin entry once the catalog has a skill), the root `README.md` + `README.zh.md` catalog tables and plugin-install lines, `.github/labels.json`, both Issue Forms' `Catalog` options, and `CATALOG_LABELS` in `scripts/sync_issue_metadata.py` |
| Any `README.md` | The matching `README.zh.md` (and vice versa) |
| `.github/labels.json` | After explicit authorization, dry-run `python3 scripts/sync_labels.py --file .github/labels.json --repo ryan-minato/skills`, then re-run with `--apply`; never `--prune` |
| `scripts/sync_labels.py` | Its origin, `skills/meta/meta-github-workflow/scripts/sync_labels.py` — the copy exists because public skills cannot reference repo files; keep behavior identical or record why it diverged |
| `skills/meta/meta-disposal/scripts/dispose.py` | Its copy, `skills/scaffold/scaffold-disposal/scripts/dispose.py` — the duplicate exists because public skills cannot reference each other; keep the two byte-identical, enforced by `check_disposal_script_copies()` in `scripts/validate_skills.py` |
| NGC/registry access in `skills/meta/meta-gpu-container/scripts/` | `skills/core/devcontainer-setup/scripts/list_sources.py` — both hit the same undocumented NGC search endpoint and nvcr.io token handshake; self-containment keeps the implementations independent, so when a registry change breaks one, check and fix the other |
| A catalog is removed | Also add its label to `RETIRED_CATALOG_LABELS` in `scripts/sync_issue_metadata.py` so stale labels can still be stripped, and drop its prefix from `CATALOG_NAME_PREFIXES` in `scripts/validate_skills.py` if it reserved one |
| Issue Form `Priority` or `Catalog` options | `.github/labels.json` and `scripts/sync_issue_metadata.py` mappings |
| PR template headings or required checklist | `scripts/check_pr_policy.py` validation |
| GitHub check name or command | `.agents/knowledge/github/checks.md`, its workflow, the remote ruleset, and `.agents/knowledge/github/platform-settings.md` |
| GitHub merge, protection, Actions, or security setting | `.agents/knowledge/github/platform-settings.md` and the corresponding workflow/authority rule |
| Project management or agent-authority policy | Its knowledge contract, `change-workflow`, and this entrypoint's routing line |
| Repo structure or check commands | This file and `ARCHITECTURE.md` |
