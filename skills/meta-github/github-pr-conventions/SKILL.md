---
name: github-pr-conventions
description: >
  Authors a repository's pull-request conventions: a PR template, a
  CONTRIBUTING pull-request rules section, path-based auto-labeling and
  checklist-validation workflows built from first-party actions only, and
  a generated project-level agent skill for opening and reviewing PRs in
  that repository — with an AGENTS.md section as the fallback deliverable
  when the project's harness does not support skills. Use when
  standardizing how a repository handles pull requests — "add a PR
  template", "enforce a PR checklist", "auto-label PRs", "document our
  review process", "standardize pull requests", or "create a skill for
  opening PRs in this repo".
license: Apache-2.0
---

# GitHub PR Conventions

Author a repository's pull-request conventions: template, contributing
rules, automation workflows, and a project-level skill (or AGENTS.md
section) that teaches agents to open and review PRs the way this
repository expects. This skill writes local files — only its outputs land
in the repository. Day-to-day PR operations belong to
`github-pull-requests`; issue templates and label taxonomy to
`github-issue-conventions`; commit-message rules to
`github-commit-conventions`.

## Assess the project first

Before authoring anything, inventory what the repository already has:
the existing PR template (`.github/pull_request_template.md`, a root or
`docs/` copy, or a `.github/PULL_REQUEST_TEMPLATE/` directory — adapt,
never replace wholesale without asking), `CONTRIBUTING.md` (root or
`.github/`), existing workflows in `.github/workflows/` (avoid file-name
collisions), the base branch and allowed merge methods
(`gh repo view -R O/R --json
defaultBranchRef,mergeCommitAllowed,squashMergeAllowed,rebaseMergeAllowed`
with `O/R` from `git remote get-url origin`), `AGENTS.md` / `CLAUDE.md`
for recorded conventions, and where project skills live — use
`.claude/skills/` if it exists, else `.agents/skills/` if it exists, else
plan to create `.agents/skills/`. Never invent structure parallel to what
the project already defines: build on what exists, or get the user's
explicit approval to replace it.
Done when: the inventory is written down and each deliverable below is
marked "new", "extends existing", or "replaces (approved)".

## Choose the deliverable

The default deliverable for workflow guidance is a **project-level agent
skill** in the skills directory found during assessment. When the project's
harness does not support skills, or the user prefers documentation, deliver
an `AGENTS.md` section (create the file if missing) or a standalone doc
instead. Ask the user once, before generating, and record the choice. All
other artifacts (templates, configs, workflows, validators) ship regardless
of this choice.

## PR template

Copy [assets/pull-request-template.md](assets/pull-request-template.md)
to `.github/pull_request_template.md`. Adapt the section contents with the
user, but keep the exact heading names in sync with the
checklist-validation workflow below — the workflow greps the PR body for
those headings, so a renamed heading makes every PR fail validation until
the workflow's heading list is updated to match.

If the repository genuinely has distinct PR kinds (for example release PRs
versus regular changes), put one file per kind in
`.github/PULL_REQUEST_TEMPLATE/<name>.md` and select one with
`?template=<name>.md` appended to the compare URL; otherwise ship the
single default template.

## CONTRIBUTING pull-request rules

Copy [assets/contributing-pr-section.md](assets/contributing-pr-section.md)
into `CONTRIBUTING.md`: append the section if the file exists, otherwise
create the file with it. Fill the `{{...}}` placeholders from the
assessment above. Read
[references/contributing-rules.md](references/contributing-rules.md) when
the user wants full contributing guidance (branch naming, merge strategy,
review expectations) beyond the shipped section.

## Automation

Copy three files:

| Asset | Destination |
|---|---|
| [assets/labeler-config.yml](assets/labeler-config.yml) | `.github/labeler.yml` |
| [assets/workflow-pr-labeler.yml](assets/workflow-pr-labeler.yml) | `.github/workflows/pr-labeler.yml` |
| [assets/workflow-pr-checklist.yml](assets/workflow-pr-checklist.yml) | `.github/workflows/pr-checklist.yml` |

Use first-party actions only (`actions/*`, `github/*`): workflows run
with repository permissions, so every third-party action is a
supply-chain decision — add one only on explicit user opt-in. Every label
key in `.github/labeler.yml` must already exist in the repository; create
missing ones before the first PR triggers the workflow.
`.github/labeler.yml` is the PR labeler's config; the issue labeler
(`github-issue-conventions`) uses `.github/issue-labeler.yml` — do not
merge the two files. Read
[references/automation-recipes.md](references/automation-recipes.md) when
customizing the labeler config syntax or adding more automation (title
validation, linked-issue enforcement, stale-PR handling).

Done when: all three files still parse as YAML after the placeholder
edits.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-prs.md](assets/project-skill-prs.md) to
`<skills-dir>/<repo-name>-prs/SKILL.md` and fill every `{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{REPO_NAME}}` | Repository name, lowercase, hyphens only |
| `{{OWNER_REPO}}` | `O/R` from the assessment |
| `{{DEFAULT_BRANCH}}` | Default branch from `gh repo view` |
| `{{MERGE_METHOD}}` | The repository's merge method (for example squash) |
| `{{TEMPLATE_HEADINGS}}` | The exact headings shipped in the PR template |
| `{{LABEL_PREFIXES}}` | Label prefixes in use (for example `area/`) |

For the AGENTS.md fallback, copy
[assets/agents-md-prs-section.md](assets/agents-md-prs-section.md) into
the project's `AGENTS.md` (create the file if missing) and fill the same
placeholders.

Refinement beyond the template pairs with `great-skill-writer` — if it is
not installed, install it from https://github.com/ryan-minato/skills.git:
`npx skills add ryan-minato/skills --skill great-skill-writer`.

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Deliver

Everything this skill wrote is local files — nothing is published yet. Hand
the changes to the project's normal git flow (branch, commit, review); that
flow, not this skill, publishes them and carries its own review gates.
Done when: the user has the list of every file created or changed, one line
each on what it does, and any follow-up steps (missing labels to create,
branch protection to enable, the first PR to watch the workflows on).

## Gotchas

- PR templates only take effect once merged to the default branch —
  testing from a feature branch shows nothing on new PRs.
- Template changes do not affect already-open PRs; a PR's body is
  snapshotted from the template at creation time.
- `actions/labeler` requires the `pull_request_target` trigger to label
  fork PRs (plain `pull_request` gets a read-only token on forks). That
  workflow must never check out or run PR code, and its permissions stay
  minimal — it runs with repository permissions.
- labeler v6 keeps the v5 config syntax (`any-glob-to-any-file` and
  friends); older v4-era configs with bare glob lists do not work.
- Renaming a heading in the PR template makes the checklist workflow
  fail every PR until its heading list is updated to match (the same
  rule the PR-template section states — one heading list, two files).
