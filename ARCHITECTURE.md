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
  knowledge/                     Contracts and conditional project knowledge
.claude/skills -> ../.agents/skills
.claude-plugin/marketplace.json  Plugin marketplace (one plugin per catalog)
.github/                         GitHub collaboration policy and automation
  skills -> ../.agents/skills    Copilot-facing path to the same skills
scripts/                         Repository tooling
justfile                         Canonical check recipes
.gitmessage                      Commit message template
```

## Catalogs

Public skills are grouped into catalogs under `skills/`:

- `core` — skills recommended for global (user-level) installation.
- `engineering` — general programming methodology skills, plus narrowly
  scoped artifact-authoring workflows (e.g. Dev Container Features,
  Templates, image prebuilds, and durable visual-design specifications)
  that do not warrant a catalog of their own.
- `meta` — disposable, project-scoped builders for generic, pluggable harness
  machinery: complete agent harness architecture, GitHub and GitLab lifecycle
  workflows, GPU container environments, and Python convention defaults.
  Install the whole catalog for a harness build. Their durable output lives
  in the target project; the builders are removed after verification.
- `scaffold` — disposable, project-scoped builders for a project of a specific
  topic (ML, data science, Colab notebooks). They are alternatives to one another: install the
  one matching the project, alongside whichever `meta` builders apply. Removed
  after verification, normally together with `meta`.
- `util` — process skills that act on the collaboration itself: clarifying,
  questioning, deciding, and organizing ideas. They are independent of domain
  and tech stack, produce a shared understanding rather than files, and keep
  their working material out of the user's workspace.
- `writing` — human-audience writing skills: genre skills (academic,
  blog/opinion, promotional copy) and medium skills (LaTeX, Typst,
  Markdown source); the general baseline `human-writing` lives in `core`.

Adding a catalog requires: the catalog scaffold (`README.md`, `README.zh.md`,
`CONTEXT.md`), an entry in this list, and — once it has a skill — a plugin
entry in `.claude-plugin/marketplace.json`. `scripts/validate_skills.py`
cross-checks this list against the directories in `skills/`.

The two disposable catalogs reserve a name prefix for their own members: every
skill in `meta` is named `meta-*` and every skill in `scaffold` is named
`scaffold-*`, enforced by `CATALOG_NAME_PREFIXES` in
`scripts/validate_skills.py`. The prefix groups a catalog's builders and
matches the marker their descriptions carry, which matters because installed
skills are read far from this repository.

The prefix is a grouping convention, not an exclusive claim on the word, and
the check runs in one direction only: `core/meta-harness` is durable, globally
installed, and `meta-`-prefixed. What identifies a disposable builder is the
marker its description opens with — that is what each catalog's disposal skill
matches on, and the only key it may use.

## Skill Visibility (symlink mechanism)

`.agents/skills/` is the canonical directory agents scan for this repo's
usable skills (the cross-client convention from the Agent Skills spec). It
contains two kinds of entries:

- **Project-only workflow skills** (`change-workflow`, `skill-authoring`,
  `code-review`): real directories, created directly here. They serve this
  repo's own workflows, are never distributed, and may reference repo paths.
- **Symlinks to public skills**: every `skills/<catalog>/<name>/` gets a
  relative symlink `.agents/skills/<name> -> ../../skills/<catalog>/<name>`,
  so the repo can dogfood the skills it publishes.

Claude Code only scans `.claude/skills/` for project skills, so
`.claude/skills` is a directory symlink to `.agents/skills`. GitHub Copilot
code review expects its review skill at `.github/skills/code-review/`, so
`.github/skills` is a directory symlink to `.agents/skills` as well. Other
clients (Codex, and anything following the spec) scan `.agents/skills/`
directly.

`scripts/validate_skills.py` (via `just validate` and pre-commit) enforces
that symlinks exist, don't dangle, and point to the right targets.

## Plugin Marketplace

`.claude-plugin/marketplace.json` publishes the repo as a Claude Code
**plugin marketplace**: one plugin per non-empty catalog. Each entry uses a
marketplace-root `source` (`"./"`) and a `skills` array that **enumerates the
catalog's skill directories** (`./skills/<catalog>/<skill>`). Those exact
paths do double duty: Claude Code loads each as a single skill (and with a
marketplace-root source the explicit list *replaces* the default scan, so a
plugin loads only its own catalog), and the `npx skills add` picker groups
skills under the catalog name by matching each path to a discovered skill.
No skill files move. Project-only skills live in `.agents/skills/` (marked
`metadata.internal: true`) and are excluded.

`scripts/gen_marketplace.py` (via `just gen-marketplace`) regenerates the
`skills` arrays from the catalogs on disk, and the validator fails if any
plugin's list drifts from its catalog — so the lists are never hand-edited.

Users add the marketplace once, then install catalogs individually:

```
/plugin marketplace add ryan-minato/skills
/plugin install <catalog>@ryan-minato-skills
```

Because installed skills are copied out of this repo, public skills must be
fully self-contained (rules in `.agents/knowledge/skill-quality.md`).

## Knowledge Base

`.agents/knowledge/*.md` is the project knowledge base. It is versioned and
reviewed with the repository; no external tracker or document service mirrors
it.

Source of truth: **the knowledge files on origin's latest default branch**.
Working-tree edits become authoritative only after merge.

The knowledge base is layered. `project-workflow.md` defines platform-neutral
management semantics, and `agent-authority.md` defines the H1 human-agent
boundary. GitHub-specific check and settings registers live under
`.agents/knowledge/github/`. `AGENTS.md` routes agents to each file only when
its decisions apply.

## GitHub Workflow

GitHub Issues provide optional task context, while every tracked change uses a
dedicated branch and pull request. `.github/labels.json` is the label taxonomy;
Issue Forms collect priority and catalog metadata, and GitHub Actions keeps
those managed labels aligned. The project-only `change-workflow` skill owns
tooling checks, explicit remote authorization, atomic commits, and the draft to
ready review lifecycle. It consumes the workflow and authority contracts rather
than defining their policy again.

The human maintainer owns integration. Same-repository pull requests use rebase
merge and fork pull requests use squash merge. GitHub cannot enforce that
conditional choice, so the target remote configuration disables merge commits
while the project skill and contributor guidance enforce the remaining
convention. `.agents/knowledge/github/platform-settings.md` records whether
that target has been synchronized.

## Quality Gates

- `just check` = skill and harness validation + `lint` (ruff over `scripts/`)
  + `pre-commit run --all-files` (whitespace, secrets scanning, ruff,
  validators). Tests written to develop harness and CI scripts are run during
  authoring but are not retained in the repository.
- pre-commit hooks are installed by `just setup` (run automatically by the
  devcontainer's `postCreateCommand`), which also sets the `.gitmessage`
  commit template.
- CI runs `just check` as `checks / quality`, validates pull-request structure
  and same-repository commit subjects as `pr / policy`, and retains the
  existing `scan-secrets` workflow. All three are intended remote required
  checks; the platform-settings register records current enforcement. CodeQL
  remains advisory.

Longer custom logic belongs in `scripts/`, not inline in justfile recipes or
hooks.
