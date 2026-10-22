# AGENTS.md

Agent entrypoint for this repository. Read this before making changes.

## Purpose

This repository is a library of [Agent Skills](https://agentskills.io):
self-contained skill directories that users install into their agents with
`npx skills add ryan-minato/skills`. Public skills live under
`skills/<catalog>/<skill-name>/`. It is long-lived, maintained by one
person (`ryan-minato`) with agents, and hosted only on GitHub.

## Repository Map

- `skills/<catalog>/<skill-name>/` — public skills; each catalog has
  `README.md`, `README.zh.md`, and `CONTEXT.md` (catalog rules).
- `.agents/skills/` — project skills (`change-workflow`, `skill-authoring`,
  `code-review`), the generated `openspec-*` skills, and symlinks to every
  public skill; `.claude/skills` and `.github/skills` link to it.
- `.agents/knowledge/` — the knowledge base; every file is listed below.
- `openspec/` — `openspec/specs/` (what public skills do), `openspec/changes/`, the project schema.
- `.github/` — issue forms, PR template, workflows, actions, labels;
  `scripts/` — management scripts in Deno; `justfile` — the commands.
- `ARCHITECTURE.md` — the mechanisms and the as-built map.

## Always

A rule that names a file in `.agents/knowledge/` is owned there, in full;
a rule that names none is owned here.

- **Language**: harness files, skills, code, and comments are English.
  Every `README.md` has a content-identical `README.zh.md` beside it.
- **Commits**: Conventional Commits in English, `type(scope): subject`;
  scope is the skill name(s), `", "`-separated, omitted for repo-wide
  changes. For distributable skills, `fix` corrects wrong, misleading, or
  wrongly restrictive or permissive installed behavior; `feat` adds a
  capability; `refactor` restructures without changing behavior; `docs`
  only for supporting documentation that changes no installed skill.
  Correction beats improvement; file format never decides the type.
- **Self-containment**: a public skill references nothing outside its
  directory; handoffs go through `ryan-minato-skills-installing` (skill-quality.md).
- **Specs before behavior**: a public skill's behavior changes through an
  OpenSpec change whose delta scenarios (WHEN/THEN) are its acceptance
  criteria; a repository change is `skip_specs`. Draft timing, approval,
  and the freeze: spec-workflow.md.
- **Checks**: run through `just` recipes; `just check` is what CI runs (github-checks.md).
- **Publishing**: nothing reaches GitHub without the user's authorization in
  this conversation and a `SAFE TO PUBLISH: YES` review (agent-authority.md `## Publishing`).
- **Authority**: agents open and ready pull requests; approving, merging,
  the approval label, settings, and widening agent authority are the
  maintainer's; never weaken a check (agent-authority.md).
- **Blocked work**: finish what does not depend on the decision, then record
  each pending decision in the handoff (agent-authority.md `## Pending decisions`).
- **Mismatches**: a document, script, or command that disagrees with the
  repository is fixed in the same change when small and certain, else
  queued as a pending decision (out of scope: an issue labelled
  `catalog/repository`, filed once authorized); never left in conversation.
- **Builders stay**: the `meta` and `scaffold` skills are this repository's
  product; their "never commit, delete after the build" rules do not apply
  here (`## Use inside this repository` in each catalog's `CONTEXT.md`).

## When To Read What

- Any tracked change (branch, issue, commits, PR, ready) → the
  `change-workflow` project skill; it runs the OpenSpec loop and the publish gate.
- Branch names, merge method, labels, milestones, triage → `.agents/knowledge/github-workflow.md`.
- A change to a skill's behavior or to the repository, anything under
  `openspec/` → `.agents/knowledge/spec-workflow.md`.
- Ready, merge, permissions, any authority question → `.agents/knowledge/agent-authority.md`.
- Creating, modifying, moving, or removing a skill → the `skill-authoring`
  project skill (bar: `.agents/knowledge/skill-quality.md`); a review → `code-review`.
- A script, recipe, hook, workflow step, or tool version → `.agents/knowledge/management-code.md`.
- A red or changed CI check → `.agents/knowledge/github-checks.md`.
- A repository setting, ruleset, security feature, or Actions policy →
  `.agents/knowledge/github-settings.md`, the declaration.
- A mirrored file, the first PR of a quarter → `.agents/knowledge/harness-maintenance.md`.
- External documentation URLs → `.agents/knowledge/references.md`.

## Validation

| Check | Command |
|---|---|
| Everything CI runs | `just check` |
| Skill layout, harness sync, catalogs, budgets | `just validate` |
| One skill while drafting | `just check-skill <dir>...` |
| Management code tests | `just test` |
| Deno lint, format, type check; ruff on skills' Python | `just lint` |
| Specs and changes (strict) | `just spec-validate` |
| Package approval state of a pull request (read-only) | `just spec-approval <n>` |
| Review threads of a pull request (read-only) | `just review-threads <n>` |
| Format and validate the tf skill assets | `just tf-validate` |
| Regenerate `marketplace.json` | `just gen-marketplace` |
| Regenerate the `openspec-*` skills after a bump of `openspec_version` | `just spec-sync` |
| Staged-change safety gate (also a pre-commit hook) | `just commit-gate` |

## Keep In Sync

Mechanically checked pairs live in `scripts/validate_skills.ts` and
`scripts/validate_harness.ts`; the rest, with owners and triggers, in
`.agents/knowledge/harness-maintenance.md`. The three that bite most often:

| When this changes | Update |
|---|---|
| Public skill added/removed | Symlink in `.agents/skills/`, catalog `README.md` + `README.zh.md`, `just gen-marketplace` |
| Any `README.md` | The matching `README.zh.md` (and vice versa) |
| Repo structure or check commands | This file and `ARCHITECTURE.md` |
