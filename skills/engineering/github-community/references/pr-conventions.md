# PR conventions: template, review rules, automation

Loaded when the task standardizes pull requests. Inputs from "Assess the
project first": `O/R`, the existing PR template inventory,
`CONTRIBUTING.md`, and the allowed merge methods
(`gh repo view -R O/R --json
defaultBranchRef,mergeCommitAllowed,squashMergeAllowed,rebaseMergeAllowed`).

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
create the file with it (placement and the file's other sections belong
to [health-files.md](health-files.md)). Fill the `{{...}}` placeholders
from the assessment. Read
[contributing-rules.md](contributing-rules.md) when the user wants full
contributing guidance (branch naming, merge strategy, review
expectations) beyond the shipped section.

## Automation

Copy three files:

| Asset | Destination |
|---|---|
| [assets/pr-labeler-config.yml](assets/pr-labeler-config.yml) | `.github/labeler.yml` |
| [assets/workflow-pr-labeler.yml](assets/workflow-pr-labeler.yml) | `.github/workflows/pr-labeler.yml` |
| [assets/workflow-pr-checklist.yml](assets/workflow-pr-checklist.yml) | `.github/workflows/pr-checklist.yml` |

Every label key in `.github/labeler.yml` must already exist in the
repository; create missing ones before the first PR triggers the
workflow. `.github/labeler.yml` is the PR labeler's config; the issue
labeler uses `.github/issue-labeler.yml` — do not merge the two files.
Read [pr-automation.md](pr-automation.md) when customizing the labeler
config syntax or when the user opts into more automation (title
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

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Gotchas

- Template changes do not affect already-open PRs; a PR's body is
  snapshotted from the template at creation time.
- `actions/labeler` requires the `pull_request_target` trigger to label
  fork PRs (plain `pull_request` gets a read-only token on forks). That
  workflow must never check out or run PR code, and its permissions stay
  minimal — it runs with repository permissions.
- labeler v6 keeps the v5 config syntax (`any-glob-to-any-file` and
  friends); older v4-era configs with bare glob lists do not work.
- Renaming a heading in the PR template makes the checklist workflow
  fail every PR until its heading list is updated to match — one heading
  list, two files.
