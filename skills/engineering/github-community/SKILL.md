---
name: github-community
description: >
  Community and convention authoring for a GitHub repository — issue forms, label
  taxonomy, PR template and review rules, commit-message conventions, versioning and
  release-notes policy (release.yml), CI validation workflows, and community health
  files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT, GOVERNANCE, FUNDING.yml,
  and the org-wide .github default repository. Use when standardizing how issues are
  filed, PRs are opened and reviewed, commits are written, or releases are versioned
  and tagged ("add a PR template", "define our labels", "set up conventional
  commits", "enforce commit format in CI"); when adding a code of conduct,
  security policy, or support/funding/governance file, or
  completing the community profile checklist; when a repo opens to outside
  contributors and needs structured intake and written rules; or when generating
  project-level skills for these workflows. Not for performing the operations
  (github-ops), nor for GitLab (gitlab-community).
license: Apache-2.0
compatibility: >
  Bundled scripts require Python 3.9+ (stdlib only); sync_labels.py needs
  an authenticated gh CLI; analyze_history.py and the shipped
  check_commits.py need git.
---

# GitHub Community

Author the files that define how a repository's collaboration works:
issue forms and a label taxonomy, the PR template and review rules,
commit-message conventions with CI enforcement, versioning and
release-notes policy, community health files, and project-level skills
that teach agents to follow all of it. This skill writes **local files
only** — the project's normal git flow publishes them. Performing the
operations (filing issues, opening PRs, cutting releases) belongs to
`github-ops`; GitLab projects to `gitlab-community`. If either is needed
and not installed, install it from
https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill github-ops

## Assess the project first

Before authoring anything, inventory what the repository already has:

- `.github/` artifacts this task creates or replaces: `ISSUE_TEMPLATE/`
  (forms, legacy `.md` templates, `config.yml`), the PR template (also
  root or `docs/` copies), workflows in `.github/workflows/` (file-name
  collisions), `.github/release.yml`.
- Existing labels: derive `O/R` from `git remote get-url origin`, then
  `gh label list -R O/R --json name,color,description` (or the MCP tool
  that lists repository labels).
- Existing tags and releases (release work):
  `git tag --sort=-v:refname | head -20`,
  `gh release list -R O/R --limit 5`.
- Commit history style (commit work): run
  [scripts/analyze_history.py](scripts/analyze_history.py) —
  `python3 scripts/analyze_history.py --max 500` prints one JSON object
  with title styles, type/scope frequencies, subject lengths, and
  trailer keys.
- Community health files at every level GitHub reads: `.github/`, the
  repo root, `docs/` — **and the account's public `.github` repository**,
  which provides org/user-wide defaults for any repo lacking its own file
  (a file may already be inherited from there).
- `CONTRIBUTING.md` (root, `.github/`, or `docs/`), `AGENTS.md` /
  `CLAUDE.md` for recorded conventions, and where project skills live —
  use `.claude/skills/` if it exists, else `.agents/skills/` if it
  exists, else plan to create `.agents/skills/`.

Never invent structure parallel to what the project already defines:
build on what exists, or get the user's explicit approval to replace it.
Done when: the inventory is written down and each deliverable is marked
"new", "extends existing", or "replaces (approved)".

## Choose the deliverable

The default deliverable for workflow guidance is a **project-level agent
skill** in the skills directory found during assessment. When the project's
harness does not support skills, or the user prefers documentation, deliver
an `AGENTS.md` section (create the file if missing) or a standalone doc
instead. Ask the user once, before generating, and record the choice. All
other artifacts (templates, configs, workflows, validators, health files)
ship regardless of this choice.

## Route by task

Read the file for the branch the task is on — now, before authoring —
and only that file:

| When the task standardizes | Read |
|---|---|
| Issue intake: forms, the label taxonomy, triage automation | [references/issue-conventions.md](references/issue-conventions.md) |
| Pull requests: template, CONTRIBUTING PR rules, PR automation | [references/pr-conventions.md](references/pr-conventions.md) |
| Commit messages: convention doc, validator, CI enforcement | [references/commit-conventions.md](references/commit-conventions.md) |
| Releases: versioning/tag policy, release.yml, tag CI check | [references/release-conventions.md](references/release-conventions.md) |
| Community health files (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT, GOVERNANCE, FUNDING), the `.github` default repo, or the community profile | [references/health-files.md](references/health-files.md) |

Ordering when several domains run together: issue conventions before
release conventions (release.yml categories key on the label taxonomy),
and commit conventions before release conventions when the version bump
rule is mapped to commit types. A CONTRIBUTING request starts in
health-files.md — it owns file placement and the non-PR sections, and
hands the PR and commit sections to their domains.

## Shared rules

- **First-party actions only** (`actions/*`, `github/*`): workflow code
  runs with the repository's permissions, so every third-party action is
  a supply-chain decision — add one only on explicit user opt-in, pinned
  to a commit SHA. Shipped CI validators are stdlib-only python3
  committed into the target repository.
- **Generated project skills are products**: same quality bar as a
  published skill — frontmatter `name` equals the directory name, and
  zero leftover `{{...}}` placeholders survive delivery. Placeholders in
  shipped assets use `{{UPPER_SNAKE}}`. Refinement beyond the templates
  pairs with `great-skill-writer`
  (`npx skills add ryan-minato/skills --skill great-skill-writer`).
- MCP tools, where mentioned, are described by capability, never by name.

## Deliver

Everything this skill wrote is local files — nothing is published yet. Hand
the changes to the project's normal git flow (branch, commit, review); that
flow, not this skill, publishes them and carries its own review gates.
Done when: the user has the list of every file created or changed, one line
each on what it does, and any follow-up steps (labels to create first,
secrets to set, branch protection to enable, the first PR or tag to watch
the workflows on).

## Gotchas (cross-domain)

- Templates, forms, `config.yml`, and workflows take effect only after
  they are merged to the repository's default branch; a feature branch
  shows nothing.
- Labels referenced by issue forms, `.github/labeler.yml`, or
  `.github/release.yml` are dropped silently when they do not exist in
  the repository — sync the taxonomy first.
- Label colors are 6-digit hex WITHOUT `#` in gh and API contexts
  (`--color d73a4a`); the leading `#` from the web UI is not accepted.
- `.github/labeler.yml` (PR labeler) and `.github/issue-labeler.yml`
  (issue labeler) are different files for different actions — merging
  them breaks both.
- A repo-local community health file always overrides the account's
  `.github` default repository; within a repo the lookup order is
  `.github/` > root > `docs/`.
- Policy documents take effect socially — the shipped CI checks are the
  only hard enforcement; say so to the user rather than implying more.
