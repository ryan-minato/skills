# Issue conventions: description templates, scoped labels, triage

Loaded when the task standardizes issue intake. Inputs from "Assess the
project first": `HOST`, `PROJECT_PATH`, the `.gitlab/issue_templates/`
inventory, and the label listing (with inherited group labels noted).

## Description templates

Copy [assets/issue-template-bug.md](assets/issue-template-bug.md),
[assets/issue-template-feature.md](assets/issue-template-feature.md),
and [assets/issue-template-default.md](assets/issue-template-default.md)
to `.gitlab/issue_templates/Bug.md`, `.gitlab/issue_templates/Feature.md`,
and `.gitlab/issue_templates/Default.md`; then edit the prompts to fit
the project. GitLab templates are plain markdown — no forms schema, no
required fields; prompts live in HTML comments, and the trailing
quick-action lines (`/label ~"type::bug" ~"status::needs-triage"`) are
what applies labels on submission.

A `/label` naming a label that does not exist is **silently ignored** —
sync the taxonomy (next section) before or together with the templates.

Read [template-authoring.md](template-authoring.md) when authoring or
editing a template beyond the shipped assets, or configuring default
templates, and a mechanism detail is uncertain (default-branch
activation, `Default.md` casing, quick-action embedding).

Done when: the templates exist and every label their quick actions name
appears in the taxonomy.

## Label taxonomy

Start from [assets/labels.json](assets/labels.json) — twelve labels on
three scoped axes (`type::`, `priority::`, `status::`) — and adjust
names, colors, and descriptions to the project with the user. The `::`
scope gives native one-per-axis exclusivity on Premium/Ultimate; on Free
the same names work as plain labels (the generated project skill
enforces one-per-axis manually), and upgrade cleanly later.

Apply it with [scripts/sync_labels.py](scripts/sync_labels.py): plan
first, validate, then execute.

```bash
python3 scripts/sync_labels.py --file labels.json --project GROUP/SUB/NAME --host HOST          # plan only
python3 scripts/sync_labels.py --file labels.json --project GROUP/SUB/NAME --host HOST --apply  # execute
```

The plan (JSON on stdout in both modes) lists create / update / in-sync,
reports inherited group labels it will never touch, and lists prune
candidates — project labels absent from the file. Pass `--prune` with
`--apply` only when the user explicitly asks: deletion strips the label
from every issue carrying it. Re-running after apply yields an
all-in-sync plan. `--group GROUP` syncs a group-level taxonomy instead.

## Automation

The templates' quick actions **are** the primary automation: they apply
the taxonomy at submission time with zero infrastructure, on any tier
and any host. GitLab has no issue-event pipelines, so there is no
GitLab-native equivalent of an issue-labeler workflow to ship.

Read [issue-automation.md](issue-automation.md) when the user opts into
more (a scheduled triage sweep for unlabeled issues, stale handling) —
those need an API token with tier-dependent options, spelled out there.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-issues.md](assets/project-skill-issues.md) to
`<skills-dir>/<project-name>-issues/SKILL.md` and fill every
`{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{PROJECT_NAME}}` | Project name, lowercase, hyphens only |
| `{{PROJECT_PATH}}` / `{{GITLAB_HOST}}` | From the origin remote |
| `{{TEMPLATES}}` | The template files and their display names |
| `{{LABEL_AXES}}` | The axes actually synced |

For the AGENTS.md fallback, copy
[assets/agents-md-issues-section.md](assets/agents-md-issues-section.md)
into the project's `AGENTS.md` (create the file if missing) and fill the
same placeholders.

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Gotchas

- `glab label edit` identifies labels by numeric `--label-id`, not name
  — use the sync script rather than hand edits.
- Quick actions: one per line, executed with the submitter's
  permissions; unknown labels silently ignored; users can delete the
  lines before submitting — they are defaults, not enforcement.
- Scoped-label exclusivity (and the two-tone rendering) is
  Premium/Ultimate; Free shows plain labels literally named
  `type::bug`, which still sort and filter fine.
- Label lists include ancestor-group labels (`is_project_label: false`);
  the project endpoints cannot edit or delete them — group-level changes
  go through the group (or the script's `--group` mode with group
  permissions).
