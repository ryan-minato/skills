# Issue conventions: forms, labels, triage automation

Loaded when the task standardizes issue intake. Inputs from "Assess the
project first": `O/R`, the existing `ISSUE_TEMPLATE/` inventory, and the
existing labels.

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

Read [issue-forms-schema.md](issue-forms-schema.md) when authoring or
editing a form beyond the shipped assets and a schema element or key is
uncertain.

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

Read [issue-automation.md](issue-automation.md) when the user opts into
automation beyond the shipped labeler (stale handling, auto-assign,
form-completeness checks).

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

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Gotchas

- `blank_issues_enabled: false` still shows a blank-issue option to users
  with write access; it only removes it for outside contributors.
- Issue-form `labels:` are applied without validation — a label that does
  not exist in the repository is dropped silently, with no error anywhere.
