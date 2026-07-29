# MR conventions: template, review rules, checklist CI

Loaded when the task standardizes merge requests. Inputs from "Assess
the project first": `HOST`, `PROJECT_PATH`, the
`.gitlab/merge_request_templates/` inventory, `CONTRIBUTING.md`, the
pipeline style (classic branch pipelines or `rules:`-based), and the
merge settings (`default_branch`, `merge_method`, `squash_option`).

## MR description template

Copy [assets/mr-template-default.md](assets/mr-template-default.md) to
`.gitlab/merge_request_templates/Default.md` and adjust the prompts.
`Default.md` auto-populates every new MR (GitLab ≥ 14.8, Free) — no URL
parameter or chooser needed; additional named templates in the same
directory appear in the template dropdown. Keep the headings exactly in
sync with the checklist job's heading list (next section) — renaming one
silently breaks the check.

Done when: the template exists and its headings match the checklist
job's list.

## CONTRIBUTING merge-request rules

Copy or append
[assets/contributing-mr-section.md](assets/contributing-mr-section.md)
into `CONTRIBUTING.md` (placement and the file's other sections belong
to [community-files.md](community-files.md)) and fill
`{{DEFAULT_BRANCH}}`, `{{BRANCH_PREFIX}}`, `{{MERGE_METHOD}}`,
`{{SQUASH_OPTION}}`, and `{{REVIEW_RESPONSE_EXPECTATION}}` from the
assessment. Read [contributing-rules.md](contributing-rules.md) for the
full guidance behind each rule (merge-method trade-offs, draft workflow,
approvals and CODEOWNERS with their tier gates).

## Checklist validation in CI

Copy the job in [assets/mr-checklist-job.yml](assets/mr-checklist-job.yml)
**directly into `.gitlab-ci.yml`** (create the file with just this job
if absent) — GitLab ignores merge-request-pipeline rules that exist only
inside `include:`d files. The job runs in merge request pipelines
(Free), validates `$CI_MERGE_REQUEST_DESCRIPTION` with POSIX shell only
(no token, no external image — safe on fork MRs), checks every required
heading, requires the security checkbox to be present AND ticked, and
fails closed when the description is truncated past the CI-variable
limit.

Two integration decisions:

- **Duplicate pipelines**: if the assessment found classic branch
  pipelines, adding an MR-event job makes every push to an MR branch
  spawn two pipelines. Offer the `workflow:rules` switch-over block in
  [mr-automation.md](mr-automation.md) — it changes when every job runs,
  so apply it only with the user's explicit agreement.
- **Making it blocking**: the failing job blocks merges only when
  Settings > Merge requests > "Pipelines must succeed" is enabled
  (Free). Tell the user to flip it; the skill cannot do it for them
  without project settings access.

Validate the edited `.gitlab-ci.yml` with `glab ci lint` when glab is
authenticated for the host.

Done when: the job is in `.gitlab-ci.yml`, the YAML parses (or `glab ci
lint` passes), and the heading list matches the template.

## Automation beyond the checklist

Read [mr-automation.md](mr-automation.md) when the user opts into more:
linked-issue or title checks inside the same job (tokenless), path-based
auto-labeling (token required, tier notes inside), stale-MR sweeps, or
the duplicate-pipeline switch-over.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-mrs.md](assets/project-skill-mrs.md) to
`<skills-dir>/<project-name>-mrs/SKILL.md` and fill every
`{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{PROJECT_NAME}}` | Project name, lowercase, hyphens only |
| `{{PROJECT_PATH}}` / `{{GITLAB_HOST}}` | From the origin remote |
| `{{DEFAULT_BRANCH}}` / `{{MERGE_METHOD}}` / `{{SQUASH_OPTION}}` | From the assessment |
| `{{TEMPLATE_HEADINGS}}` | The exact headings shipped in the MR template |

For the AGENTS.md fallback, copy
[assets/agents-md-mrs-section.md](assets/agents-md-mrs-section.md) into
the project's `AGENTS.md` (create the file if missing) and fill the same
placeholders. The skill template pre-wires the host-checked glab path,
the project's conventions, the condensed pre-publish gate, and
create/review tables.

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Gotchas

- `$CI_MERGE_REQUEST_DESCRIPTION` is capped (2700 characters, GitLab
  ≥ 16.7, with `$CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED` set when
  cut); the job fails closed on truncation — keep the template compact.
  On older instances the truncation flag is unset, which the job treats
  as not-truncated.
- Renaming a template heading without updating the job's heading list
  silently breaks validation — they must move together.
- Fork MRs run the MR pipeline in the fork's context: the shipped job is
  tokenless by design — never add secrets to it.
- CODEOWNERS and required approval rules are Premium/Ultimate; on Free
  the CODEOWNERS file is inert and approvals are optional — CONTRIBUTING
  text carries the review rules there.
