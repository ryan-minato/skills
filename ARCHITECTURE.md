# Architecture

How this repository's mechanisms fit together. For day-to-day conventions,
start at [AGENTS.md](AGENTS.md).

## Layout

```
skills/<catalog>/<skill-name>/   Public, distributable skills
  <catalog>/README.md            What the catalog is for + skill table (EN)
  <catalog>/README.zh.md         Content-identical Chinese translation
  <catalog>/CONTEXT.md           Catalog-scoped rules and reference URLs
.agents/
  skills/                        Skills visible to this repo's agents
  knowledge/                     Rules and registers, each with a pointer in AGENTS.md
openspec/                        Specifications: specs/ (source of truth), changes/, changes/archive/
.claude/skills -> ../.agents/skills
.claude/settings.json            Claude Code permission entries (allow, deny)
.claude-plugin/marketplace.json  Plugin marketplace (one plugin per catalog)
.codex/config.toml               Codex approvals, sandbox, agentskills MCP server
.codex/rules/default.rules       Codex command rules (forbidden, allow)
.mcp.json                        agentskills MCP server (Claude Code)
.devcontainer/                   Dev container: latest tools, MCP server, `just setup` on create
.github/                         Issue forms, PR template, workflows, labels, health files
  actions/setup-tools/           Composite action: latest Deno, other tools on request
  skills -> ../.agents/skills    Copilot-facing path to the same skills
scripts/                         Management scripts (TypeScript run by Deno) and their tests
  lib/                           Modules the scripts share; no remote import
deno.json                        Deno import map, no lockfile, fmt/lint/test scope
justfile                         Canonical recipes and the one tool pin (OpenSpec)
ruff.toml                        Lint and format configuration for the public skills' Python
.pre-commit-config.yaml          Hooks: commit safety, secrets, ruff, validators, no commit on main
.gitmessage                      Commit message template
```

This file is also the as-built map for everything `openspec/specs/` does
not cover: descriptive, never normative. Behavior that a spec describes is
ruled by the spec.

## Catalogs

Public skills are grouped into catalogs under `skills/`:

- `core` — skills recommended for global (user-level) installation.
- `engineering` — general programming methodology skills, plus narrowly
  scoped artifact-authoring workflows (e.g. Dev Container Features,
  Templates, image prebuilds, and durable visual-design specifications)
  that do not warrant a catalog of their own. It also holds the brownfield
  suite — five skills prefixed brownfield- for understanding, specifying,
  and migrating existing codebases — the catalog's only members granted
  dependencies on one another.
- `sdd` — durable skills for spec-driven development: the framework-
  agnostic methodology (`spec-driven-development`) and one skill per
  framework (`openspec-workflow`, `spec-kit-workflow`) carrying that
  tool's approval package, in-request archive or completion, comment
  commands, status labels, and per-platform automation.
- `meta` — disposable, project-scoped builders for generic, pluggable harness
  machinery: one entry workflow (`meta-harness-building`) that routes to an
  architecture manual, contract builders (git branching, project workflow,
  agent authority) that design platform-neutrally and deposit
  platform-worded decisions, a paradigm builder (specification workflow)
  that deposits its contract and later fills the platform base's
  extension slots, paradigm-neutral GitHub and GitLab lifecycle bases that
  implement the deposits and register those slots, GPU container
  environments, Python convention defaults, and the disposal step. Install
  the whole catalog for a harness build. Their durable output lives in the
  target project; the builders are removed after verification.
- `scaffold` — disposable, project-scoped builders for a project of a specific
  topic (ML, data science, Colab notebooks). They are alternatives to one another: install the
  one matching the project, alongside whichever `meta` builders apply. Removed
  after verification, normally together with `meta`.
- `machine-learning` — durable, per-project skills for the daily work of
  a project that trains or evaluates models: run provenance, the research
  task workflow, experiment code conventions, training instrumentation,
  and training diagnosis. Installed one at a time; siblings are handoffs by
  role. Initialization stays with `scaffold-ml`.
- `writing` — human-audience writing skills: genre skills (academic,
  blog/opinion, promotional copy) and medium skills (LaTeX, Typst,
  Markdown source); the general baseline `human-writing` lives in `core`.

A catalog is its scaffold (`README.md`, `README.zh.md`, `CONTEXT.md`), an
entry in this list, and — once it has a skill — a plugin entry in
`.claude-plugin/marketplace.json`. `scripts/validate_skills.ts`
cross-checks this list against the directories in `skills/`; the steps
for adding or removing one are in the `skill-authoring` project skill.

Skill names follow a recommended `[<prefix>-]<body>[-<suffix>]` shape,
described in `.agents/knowledge/skill-quality.md` and refined per catalog in
its `CONTEXT.md` `## Naming` section; the shape is advisory. What is enforced
is the prefix the two disposable catalogs reserve for their members: every
skill in `meta` is named `meta-*` and every skill in `scaffold` is named
`scaffold-*`, through the catalog prefix list in `scripts/validate_skills.ts`.
The prefix groups a catalog's builders, which matters because installed skills
are read far from this repository.

The prefix is a grouping convention, not an exclusive claim on the word, and
the check runs in one direction only: `core/meta-harness` is durable, globally
installed, and `meta-`-prefixed. What identifies a disposable builder is the
marker its description opens with — the same sentence in both catalogs,
`Disposable builder skill (delete after the harness is built):`, enforced by
the disposable-marker check of `scripts/validate_skills.ts`. That marker is
what `meta-disposal` matches on, and the only key it may use, so one disposal
skill removes both catalogs' builders together.

## Skill Visibility (symlink mechanism)

`.agents/skills/` is the canonical directory agents scan for this repo's
usable skills (the cross-client convention from the Agent Skills spec). It
contains three kinds of entries:

- **Project-only workflow skills** (`change-workflow`, `skill-authoring`,
  `code-review`): real directories, created directly here. They serve this
  repo's own workflows, are never distributed, and may reference repo paths.
- **OpenSpec-generated skills** (`openspec-*`, plus the
  `.agents/skills/.openspec-target` marker): real directories written by
  the OpenSpec CLI for the shared `.agents/skills` target and regenerated
  only by `just spec-sync`. Tracked, never distributed.
- **Symlinks to public skills**: every `skills/<catalog>/<name>/` gets a
  relative symlink `.agents/skills/<name> -> ../../skills/<catalog>/<name>`,
  so the repo can dogfood the skills it publishes.

Claude Code only scans `.claude/skills/` for project skills, so
`.claude/skills` is a directory symlink to `.agents/skills`. GitHub Copilot
code review expects its review skill at `.github/skills/code-review/`, so
`.github/skills` is a directory symlink to `.agents/skills` as well. Other
clients (Codex, and anything following the spec) scan `.agents/skills/`
directly.

`scripts/validate_skills.ts` (through `just validate` and pre-commit)
checks that the symlinks exist, do not dangle, and point to the right
targets.

## Plugin Marketplace

`.claude-plugin/marketplace.json` publishes the repo as a Claude Code
**plugin marketplace**: one plugin per non-empty catalog. Each entry uses a
marketplace-root `source` (`"./"`) and a `skills` array that **enumerates the
catalog's skill directories** (`./skills/<catalog>/<skill>`). Those exact
paths do double duty: Claude Code loads each as a single skill (and with a
marketplace-root source the explicit list *replaces* the default scan, so a
plugin loads only its own catalog), and the `npx skills add` picker groups
skills under the catalog name by matching each path to a discovered skill.
No skill files move. The real directories in `.agents/skills/` (project-only
and OpenSpec-generated skills) are excluded because the generator reads
`skills/` only.

`scripts/gen_marketplace.ts` (through `just gen-marketplace`) regenerates
the `skills` arrays from the catalogs on disk, and `scripts/validate_skills.ts`
fails when a plugin's list drifts from its catalog, so the lists are never
hand-edited.

Users add the marketplace once, then install catalogs individually:

```
/plugin marketplace add ryan-minato/skills
/plugin install <catalog>@ryan-minato-skills
```

Installed skills are copied out of this repository, which is why public
skills are self-contained (`.agents/knowledge/skill-quality.md`).

## Knowledge Base

`.agents/knowledge/*.md` is the project knowledge base: rule files
(`github-workflow.md`, `spec-workflow.md`, `agent-authority.md`,
`management-code.md`), the checks register (`github-checks.md`), the
settings declaration (`github-settings.md`), the synchronization and
entropy register (`harness-maintenance.md`), the skill quality bar
(`skill-quality.md`), and external URLs (`references.md`). Each rule has
one owner file; other files point to it. `AGENTS.md` names every file with
the condition that loads it.

`scripts/validate_harness.ts` fails when a knowledge file has no pointer
in `AGENTS.md`, when a path or `just` recipe `AGENTS.md` names does not
exist, and — its `budget` check — when `AGENTS.md` passes 110 lines or a
table row in `AGENTS.md`, this file, or a knowledge file passes 200
columns.

The knowledge files on origin's latest default branch are authoritative;
working-tree edits become authoritative only after merge.

## Specifications

`openspec/specs/<catalog>/<skill-name>/spec.md` holds what each public
skill does; `openspec/changes/<slug>/` holds a change in flight and
`openspec/changes/archive/` the archived ones. Change artifacts follow
the project schema in `openspec/schemas/skill-change/` (requirements named
by kind — trigger, behavior, handoff, script — and a design that bounds
the approach and carries the verification plan). The lifecycle around
them is `.agents/knowledge/spec-workflow.md`.

The OpenSpec CLI runs through Deno at the version the `justfile` pins:
`just install-tools` installs it into the ignored .tools/bin,
`just spec-validate` runs its strict validator, `just spec-check` adds the
unarchived-change rule for the changes a branch touches, and
`just spec-sync` regenerates the `openspec-*` skills.
`scripts/spec_changes.ts` is the related-change tool behind
`just spec-changes` (`snapshot`, `status`, `show`, `check`, `archive`,
`labels`); `checks / spec`, `spec / command`, and `spec / labels` run it.

The package approval is the label `spec/approved`. The maintainer adds it;
`spec / labels` then runs `scripts/spec_approval.ts`, which decides from
the pull request's current labels and timeline, writes one record comment
as `github-actions[bot]` naming the approved head, the account, and the
label event, and withdraws the label when the package changes after the
recorded head, when a run fails, or when its deciding step times out.
`just spec-approval <n>` recomputes the state without writing. The rules
around the label are in `.agents/knowledge/spec-workflow.md`
`## Approval gates`.

## GitHub Workflow

The issue forms apply type and `status/needs-triage` labels; the
`issues / triage` workflow derives `priority/*` and `catalog/*` from the
form answers with `scripts/sync_issue_metadata.ts`, whose mapping comes
from `.github/labels.json` by prefix. The `labels / sync` workflow applies
`.github/labels.json` to the repository with `scripts/sync_labels.ts`
whenever it changes on `main` and weekly, reporting labels it would prune
instead of deleting them. `scripts/check_pr_policy.ts` checks each pull
request against the base's template and the commit convention, and
`scripts/check_commit_safety.ts --range` scans each of its commits. `scripts/review_threads.ts` (`just review-threads <n>`)
reads a pull request's review threads through one fixed GraphQL query.

Where each workflow finds its code:

- `checks / *` and `scan-secrets` run the pull request's own copy.
- `pr / policy` takes its workflow file from the pull request's merge ref
  but checks out the base and runs the base's script, so a change to its
  script takes effect from the next pull request.
- `spec / labels` (`pull_request_target`), `spec / command`
  (`issue_comment`), and `issues / triage` (`issues`) run `main`'s
  workflow and scripts; `labels / sync` runs on `main` only. A change to
  their scripts takes effect after merge.

## Checks

- `just check` runs `validate` (`scripts/validate_skills.ts`,
  `scripts/validate_harness.ts`), `lint` (Deno lint, format, and type
  check on the management code; ruff on the public skills' Python),
  `test` (`deno test`), `spec-validate`, `tf-validate` (the OpenTofu skill
  assets with the latest OpenTofu), and the remaining pre-commit hooks.
- `just setup`, run by the dev container's `postCreateCommand`, installs
  the tools (`just install-tools`), the pre-commit hooks, and the
  `.gitmessage` commit template.
- CI installs its tools with `.github/actions/setup-tools/action.yml`. The
  `checks / gate` job runs `scripts/ci_gate.ts` over the results of the
  jobs it needs; `checks / devcontainer` builds the dev container and runs
  `just check` inside it. Every job, its command, and the required checks
  are in `.agents/knowledge/github-checks.md`.
- `.agents/skills/change-workflow/scripts/run_log_digest.ts` digests a
  failed run's logs for diagnosis.
