---
name: github-issue-conventions
description: >
  Issue conventions for a GitHub repository — author and automate how its issues are
  filed, labeled, and triaged. Use when adding or overhauling issue templates or
  forms; when defining or restructuring the repository's labels, or asked what labels
  it needs; when automating issue triage or labeling; when incoming reports arrive
  unstructured or missing key details; when a repo opens to outside contributors and
  needs structured intake; or when creating a skill for filing issues in this repo.
  Not for filing, commenting on, or closing an issue (github-issues), one-off label or
  milestone edits (github-planning), or GitLab (gitlab-issue-conventions).
license: Apache-2.0
compatibility: >
  scripts/sync_labels.py requires Python 3.9+ (stdlib only) and an
  authenticated gh CLI.
---

# GitHub Issue Conventions

Author the files that define how a repository's issues are filed, labeled,
and triaged: issue forms, a label taxonomy, labeling automation, and a
project-level skill (or AGENTS.md section) that teaches agents in that
repository to follow all of it. This skill writes local files — only its
outputs land in the repository. Day-to-day issue operations belong to
`github-issues`; PR conventions to `github-pr-conventions`; commit and
release policy to `github-commit-conventions` /
`github-release-conventions`.

## Assess the project first

Before authoring anything, inventory what the repository already has:
`.github/ISSUE_TEMPLATE/` (existing forms, legacy `.md` templates,
`config.yml`), existing labels (derive `O/R` from `git remote get-url
origin`, then `gh label list -R O/R --json name,color,description` — or
the MCP tool that lists repository labels), existing issue automation in
`.github/workflows/`, `AGENTS.md` / `CLAUDE.md` for recorded conventions,
and where project skills live — use `.claude/skills/` if it exists, else
`.agents/skills/` if it exists, else plan to create `.agents/skills/`.
Never invent structure parallel to what the project already defines:
build on what exists, or get the user's explicit approval to replace it.
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

## Issue forms

Copy [assets/issue-form-bug.yml](assets/issue-form-bug.yml) and
[assets/issue-form-feature.yml](assets/issue-form-feature.yml) to
`.github/ISSUE_TEMPLATE/bug.yml` and `.github/ISSUE_TEMPLATE/feature.yml`,
and [assets/issue-template-config.yml](assets/issue-template-config.yml)
to `.github/ISSUE_TEMPLATE/config.yml`; then edit the placeholders
(project name, contact links, labels) to fit the repository.

Every label a form references must exist in the repository, or GitHub
silently drops it — sync the label taxonomy (next section) before or
together with the forms.

Read [references/issue-forms-schema.md](references/issue-forms-schema.md)
when authoring or editing a form beyond the shipped assets.

Done when: each form file parses as YAML and references only labels
present in the taxonomy.

## Label taxonomy

Start from [assets/labels.json](assets/labels.json) — twelve labels on
three axes (`type/*`, `priority/*`, `status/*`) — and adjust names,
colors, and descriptions to the repository with the user.

Apply it with [scripts/sync_labels.py](scripts/sync_labels.py): plan
first, validate, then execute.

```bash
python3 scripts/sync_labels.py --file labels.json --repo O/R          # plan only, changes nothing
python3 scripts/sync_labels.py --file labels.json --repo O/R --apply  # execute the plan
```

The plan (JSON on stdout in both modes) lists create / update / skip and
reports prune candidates — labels present in the repo but absent from the
file. Pass `--prune` together with `--apply` to delete those, and only
when the user explicitly asks: deletion strips the label from every issue
that carries it. The script is idempotent; re-running after apply yields
all-skip.

If only MCP is available (no gh), apply the printed plan one label at a
time with the MCP tool that creates or updates a label, or hand the plan
to the user.

## Automation

Copy [assets/workflow-issue-labeler.yml](assets/workflow-issue-labeler.yml)
to `.github/workflows/issue-labeler.yml` and its configuration
[assets/issue-labeler-config.yml](assets/issue-labeler-config.yml) to
`.github/issue-labeler.yml` — NOT `.github/labeler.yml`, which belongs to
the PR labeler (`actions/labeler`); a collision breaks both.

Shipped automation uses first-party actions only (`github/*` or
`actions/*`) because workflow code runs with the repository's
permissions; add a third-party action only on explicit user opt-in.

Read [references/automation-recipes.md](references/automation-recipes.md)
when adding automation beyond the shipped labeler (stale handling,
auto-assign, form-completeness checks).

Done when: the workflow and its config file still parse as YAML after
the placeholder edits.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-issues.md](assets/project-skill-issues.md) to
`<skills-dir>/<repo-name>-issues/SKILL.md` and fill every
`{{PLACEHOLDER}}`: `{{REPO_NAME}}` and `{{OWNER_REPO}}` from the origin
remote, `{{FORMS}}` with the form files and their display names,
`{{LABEL_AXES}}` with the axes actually synced. The template pre-wires
the repository's issue forms, its label taxonomy, capability-described
MCP alternatives, and the condensed pre-publish gate.

For the AGENTS.md fallback, copy
[assets/agents-md-issues-section.md](assets/agents-md-issues-section.md)
into the project's `AGENTS.md` (create the file if missing) and fill the
same placeholders.

For refinement beyond the template this pairs with `great-skill-writer`.
If it is not installed, install it from
https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill great-skill-writer

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Deliver

Everything this skill wrote is local files — nothing is published yet. Hand
the changes to the project's normal git flow (branch, commit, review); that
flow, not this skill, publishes them and carries its own review gates.
Done when: the user has the list of every file created or changed, one line
each on what it does, and any follow-up steps (secrets to set, first sync
run, branch protection to enable).

## Gotchas

- Issue forms and `config.yml` take effect only after they are merged to
  the repository's default branch; a feature branch shows nothing.
- `blank_issues_enabled: false` still shows a blank-issue option to users
  with write access; it only removes it for outside contributors.
- Issue-form `labels:` are applied without validation — a label that does
  not exist in the repository is dropped silently, with no error anywhere.
- Label colors are 6-digit hex WITHOUT `#` in gh and API contexts
  (`--color d73a4a`); the leading `#` shown in GitHub's web UI is not
  accepted there.
- The label taxonomy feeds more than issues: `.github/release.yml`
  (release-notes categories) keys on the same labels — when both this
  skill and `github-release-conventions` run, run this one first.
