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
  outside their own directory — no links to repo files and no `README.md`
  inside a skill root. Naming another skill is a dependency: any public
  skill may depend on `core` skills; dependencies inside a catalog or on
  another catalog need a grant in that catalog's `CONTEXT.md` (default:
  none); skills from other repositories are never depended on or
  recommended. Every handoff routes through the
  `ryan-minato-skills-installing` skill, never an install command. Full
  standards: `.agents/knowledge/skill-quality.md`.
- **Skill naming**: a recommended `[<prefix>-]<body>[-<suffix>]` shape in
  `.agents/knowledge/skill-quality.md`, refined per catalog in its
  `CONTEXT.md`. Advisory: only reserved catalog prefixes are enforced, and a
  name that departs from the shape is not a defect.
- **Checks**: always run checks through justfile recipes (`just check`),
  never ad-hoc equivalents, so results are consistent everywhere.
- **Commits**: Conventional Commits in English; scope is the modified skill
  name(s), `", "`-separated; omit the scope for non-skill or repo-wide
  changes. Classify changes to distributable skills by their effect: `fix`
  corrects wrong, misleading, overly restrictive, or overly permissive
  installed behavior; `feat` adds a new capability; and `refactor`
  restructures without changing behavior. Correction takes precedence over
  improvement. Use `docs` only for supporting documentation that does not
  change an installed skill; file format does not decide the type. Template:
  `.gitmessage` (installed by `just setup`).
- **Task management**: GitHub is the only task-management platform. Issues are
  optional; every change has a dedicated branch and PR. Link an Issue with
  `Closes #N`, or record `N/A — <reason>` in the PR.
- **Workflow**: before tracked work, verify `gh` installation and login and
  obtain explicit authorization for remote writes. Use atomic commits and the
  draft-to-ready PR lifecycle in the `change-workflow` project skill.

## When To Read What

- Starting any proactive change, or preparing a branch, commit series, or
  PR handoff → use the `change-workflow` project skill.
- Creating or modifying any skill → use the `skill-authoring` project skill;
  it routes to `.agents/knowledge/skill-quality.md`, the catalog's
  `CONTEXT.md`, subagent-capability-gated behavioral tests, and independent
  script and static validation.
- Catalog-specific rules and references → `skills/<catalog>/CONTEXT.md`
  (catalog-scoped material belongs there, not in the global references).
- External documentation URLs → `.agents/knowledge/references.md`.
- Labels, Issue Forms, or PR policy changed → inspect `.github/` and keep every
  referenced label and exact validated heading synchronized.
- Repo mechanics (symlinks, plugin marketplace, sync design) →
  `ARCHITECTURE.md`.

## Validation

- `just check` — everything (validator, lint, pre-commit hooks).
- `just validate` — skill layout and harness consistency only.
- `just check-skill <dir>...` — lint specific skill directories while drafting.
- `just lint` — ruff over `scripts/`.
- `just commit-gate` — pre-commit safety gate over staged changes.
- `just gen-marketplace` — regenerate `marketplace.json` skills[] after
  adding or removing a public skill.

## Keep In Sync

| When this changes | Update |
|---|---|
| Public skill added/removed | Symlink in `.agents/skills/`, catalog `README.md` + `README.zh.md`, and `.claude-plugin/marketplace.json` (`just gen-marketplace`) |
| Public skill moved or renamed | First `git grep -n <old-name>` across `skills/`, `.agents/`, `.github/`, and the root documents, and fix every reference (other skills' handoffs, catalog `CONTEXT.md` routing, project skills) in the same change; then the added/removed row above |
| A catalog `CONTEXT.md` `## Dependencies` grant | The current-grants summary in `.agents/knowledge/skill-quality.md` and the catalog's bullet in `ARCHITECTURE.md` |
| The handoff template in `.agents/knowledge/skill-quality.md` | The description triggers of `skills/core/ryan-minato-skills-installing` — the template's wording is what makes that skill fire when another skill hands off to it |
| Catalog added/removed | Catalog scaffold, the Catalogs section in `ARCHITECTURE.md`, `.claude-plugin/marketplace.json` (a plugin entry once the catalog has a skill), the root `README.md` + `README.zh.md` catalog tables and plugin-install lines, `.github/labels.json`, both Issue Forms' `Catalog` options, and the `catalogLabels` map in `.github/workflows/issue-metadata.yml` |
| Any `README.md` | The matching `README.zh.md` (and vice versa) |
| `.github/labels.json` | After explicit authorization, dry-run `python3 scripts/sync_labels.py --file .github/labels.json --repo ryan-minato/skills`, then re-run with `--apply`; never `--prune` |
| `scripts/sync_labels.py` | Its origin, `skills/meta/meta-github-workflow/scripts/sync_labels.py` — the copy exists because public skills cannot reference repo files; keep behavior identical or record why it diverged |
| `skills/meta/meta-disposal/scripts/dispose.py` | Its copy, `skills/scaffold/scaffold-disposal/scripts/dispose.py` — the duplicate exists because public skills cannot reference each other; keep the two byte-identical, enforced by `check_disposal_script_copies()` in `scripts/validate_skills.py` |
| NGC/registry access in `skills/meta/meta-gpu-container/scripts/` | `skills/core/devcontainer-setup/scripts/list_sources.py` — both hit the same undocumented NGC search endpoint and nvcr.io token handshake; self-containment keeps the implementations independent, so when a registry change breaks one, check and fix the other |
| A catalog contract's install mode (whole vs. by name) | `CATALOGS_INSTALLED_WHOLE` in `skills/core/ryan-minato-skills-installing/scripts/install_skill.py`, which mirrors the catalog `CONTEXT.md` contracts so the fallback installer refuses whole-catalog installs of catalogs whose builders are alternatives |
| A catalog is removed | Also add its label to `retiredCatalogLabels` in `.github/workflows/issue-metadata.yml` so stale labels can still be stripped, and drop its prefix from `CATALOG_NAME_PREFIXES` in `scripts/validate_skills.py` if it reserved one |
| Issue Form `Priority` or `Catalog` options | `.github/labels.json` and `.github/workflows/issue-metadata.yml` mappings |
| PR template headings or required checklist | `.github/workflows/pr-policy.yml` validation |
| Repo structure or check commands | This file and `ARCHITECTURE.md` |
