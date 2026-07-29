---
name: gitlab-community
description: >
  Community and convention authoring for a GitLab project — issue and MR description
  templates under .gitlab/, scoped-label taxonomy, commit-message rules and Changelog
  trailers, versioning and tag policy (changelog_config.yml), tokenless CI validation
  jobs, and community files (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY) as conventional
  root files. Use when standardizing how issues are filed, MRs are described and
  reviewed, commits are written, or releases are versioned ("add an MR template",
  "define our labels", "set up changelog generation"); when the user says "PR
  template" but the repository is on
  GitLab; when intake arrives unstructured and .gitlab/ templates need authoring;
  when adding a code of conduct or security policy, or documenting contributor and
  review rules in CONTRIBUTING; or when generating project-level skills for these
  workflows. Not for performing the operations (gitlab-ops), nor for GitHub
  (github-community).
license: Apache-2.0
compatibility: >
  Bundled scripts require Python 3.9+ (stdlib only); sync_labels.py needs
  a glab CLI authenticated against the target host; analyze_history.py
  and the shipped check_commits.py need git.
---

# GitLab Community

Author the files that define how a GitLab project's collaboration works:
issue and MR description templates, a scoped-label taxonomy,
commit-message conventions with tokenless CI enforcement, versioning and
changelog policy, community files, and project-level skills that teach
agents to follow all of it — on gitlab.com or any self-managed host.
This skill writes **local files only** — the project's normal git flow
publishes them (the one server action is the label sync's explicit
`--apply`). Performing the operations (filing issues, opening MRs,
cutting releases) is out of scope (the `gitlab-ops` skill's territory),
as are GitHub repositories (`github-community`).

## Assess the project first

Before authoring anything, inventory what the project already has:

- Derive `HOST` and the full `PROJECT_PATH` from `git remote get-url
  origin` (host = the part right after `https://` or the `@`; path = the
  rest minus `.git`, nesting kept whole — never assume gitlab.com).
- `.gitlab/issue_templates/` and `.gitlab/merge_request_templates/`
  (existing templates, especially a `Default.md` in any casing — adapt
  rather than replace).
- Existing labels via `glab label list -F json -P 100` run inside the
  checkout — the listing includes **inherited group labels** the project
  endpoints cannot edit; note which are which.
- `.gitlab-ci.yml`: existing job names (collisions) and whether jobs run
  as classic branch pipelines (no `rules:`/`workflow:` keys — this
  decides how MR-event jobs integrate).
- Merge settings from inside the checkout: `glab api projects/:fullpath`
  → `default_branch`, `merge_method`, `squash_option`.
- Commit history style (commit work): run
  [scripts/analyze_history.py](scripts/analyze_history.py) —
  `python3 scripts/analyze_history.py --max 500` prints one JSON object
  with title styles, type/scope frequencies, subject lengths, and
  trailer keys (a `Changelog` row means the project already feeds
  changelog generation).
- Existing tags and releases (release work):
  `git tag --sort=-v:refname | head -20`,
  `glab release list -R PROJECT_PATH`, milestone usage.
- Community files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md` — root or `docs/`), `AGENTS.md` / `CLAUDE.md` for
  recorded conventions, and where project skills live — use
  `.claude/skills/` if it exists, else `.agents/skills/` if it exists,
  else plan to create `.agents/skills/`.

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
other artifacts (templates, configs, CI jobs, validators, community files)
ship regardless of this choice.

## Route by task

Read the file for the branch the task is on — now, before authoring —
and only that file:

| When the task standardizes | Read |
|---|---|
| Issue intake: description templates, the scoped-label taxonomy, triage automation | [references/issue-conventions.md](references/issue-conventions.md) |
| Merge requests: template, CONTRIBUTING MR rules, checklist CI | [references/mr-conventions.md](references/mr-conventions.md) |
| Commit messages: convention doc, validator, MR-pipeline enforcement, Changelog trailers | [references/commit-conventions.md](references/commit-conventions.md) |
| Releases: versioning/tag policy, changelog config, milestone policy, tag CI check | [references/release-conventions.md](references/release-conventions.md) |
| Community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY | [references/community-files.md](references/community-files.md) |

Ordering when several domains run together: commit conventions before
release conventions (the changelog config consumes the trailer habit it
establishes), and issue conventions before anything keyed to its labels.
A CONTRIBUTING request starts in community-files.md — it owns file
placement and the non-MR sections, and hands the MR and commit sections
to their domains.

## Shared rules

- **Tokenless CI by default.** Shipped CI jobs run without job tokens or
  secrets (MR-event checklist/format validation, `$CI_COMMIT_TAG` tag
  checks) and are safe on fork MRs — never add secrets to them. Anything
  needing a token appears only behind an explicit user opt-in, marked as
  such. Free-tier mechanisms are the default; Premium/Ultimate features
  carry a tier badge. CI snippets write hosts as
  `$CI_SERVER_HOST`/`$CI_API_V4_URL`, never literal hostnames. Shipped
  validators are stdlib-only python3 committed into the target repo.
- **Quick actions are a feature and a hazard.** Templates deliberately
  embed quick actions (`/label ...`) as their automation; document every
  one a shipped template embeds — in the template and in the deliverable
  summary.
- **Generated project skills are products**: frontmatter `name` equals
  the directory name, and zero leftover `{{...}}` placeholders survive
  delivery. Placeholders use `{{UPPER_SNAKE}}`.
- MCP tools, where mentioned, are described by capability, never by name.

## Deliver

Everything this skill wrote is local files — nothing is published yet. Hand
the changes to the project's normal git flow (branch, commit, review); that
flow, not this skill, publishes them and carries its own review gates.
Done when: the user has the list of every file created or changed, one line
each on what it does, and any follow-up steps (label sync to run,
"Pipelines must succeed" to enable, the default-branch merge that
activates templates, the first MR or tag to watch the jobs on).

## Gotchas (cross-domain)

- Templates take effect only from the **default branch**; a feature
  branch shows nothing. Template changes affect only new items — the
  description is snapshotted at creation.
- A `/label` quick action naming a label that does not exist is
  **silently ignored** — sync the taxonomy before shipping templates
  that reference it.
- Label colors REQUIRE the leading `#` (`#d73a4a`) in GitLab's API and
  glab — the bare `d73a4a` form GitHub uses is rejected.
- MR-pipeline `rules:` for the checklist job count only when written
  directly in `.gitlab-ci.yml`; rules inside `include:`d files do not
  enable merge request pipelines (the commit-check job is exempt — its
  rules evaluate the same either way).
- Adding an MR-event job to a project with classic branch pipelines can
  spawn duplicate pipelines — offer the `workflow:rules` switch-over
  only with the user's explicit agreement.
- Shipped CI checks block merges only when Settings > Merge requests >
  "Pipelines must succeed" is enabled (Free); hard server-side commit
  enforcement is a Premium push rule.
- GitLab returns **404, not 403**, for tier-gated features — report the
  tier requirement instead of retrying.
- Policy documents take effect socially — the shipped CI checks are the
  only hard enforcement; say so to the user rather than implying more.
