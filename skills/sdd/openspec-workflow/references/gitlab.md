# Installing the automation on GitLab

Read when installing or changing the OpenSpec request automation in a
GitLab project. The assets under `assets/gitlab/` are raw shapes: every
`{{PLACEHOLDER}}` is replaced before the file lands.

## What lands where

| Piece | From | To | Notes |
|---|---|---|---|
| Script | `scripts/spec_changes.py` | the project's `scripts/spec_changes.py` | byte-identical copy |
| Jobs | `assets/gitlab/ci-spec-jobs.yml` | included from `.gitlab-ci.yml` (or pasted), stage `spec` | `spec:check`, `spec:show`, `spec:status`, `spec:archive`, `spec:labels` |
| Labels | `assets/gitlab/labels-spec.json` | the project's label file, synced with its label tool | GitLab colors carry `#`; on Premium the two axes may become scoped labels for built-in exclusivity |

Placeholders: `{{NODE_IMAGE}}`, `{{INSTALL_COMMAND}}`,
`{{VALIDATE_COMMAND}}`, `{{BOT_NAME}}`, `{{BOT_EMAIL}}`,
`{{ARCHIVE_COMMIT_PREFIX}}`, `{{ARCHIVE_COMMIT_SUBJECT}}`.

## What GitLab offers, and does not

Verified against GitLab's documentation on 2026-09-17:

- No pipeline source exists for a merge request note or a label change.
  Merge request pipelines run when the request is created, when its source
  branch receives a push, from the **Run pipeline** button, or from the
  `/run_pipeline` quick action (GitLab 18.7 and later). A label therefore
  takes effect on the next pipeline; `spec:archive` reads
  `CI_MERGE_REQUEST_LABELS` then.
- Quick actions carry no arguments a job could read, so `spec:show` and
  `spec:status` are manual jobs whose arguments come from the manual job's
  variables form (`SPEC_CHANGE`, `SPEC_DOC`); their output goes to the job
  log and, with `SPEC_NOTE_TOKEN`, to a merge request note.
- A push to the source branch needs a project access token
  (`write_repository` and `api`) in a masked CI/CD variable
  (`SPEC_ARCHIVE_TOKEN`); merge request pipelines run on unprotected source
  branches, so the variable must be unprotected and any job on any branch
  of the project can read it — scope it narrowly, rotate it, and record it
  in the settings knowledge by name only. A token push starts the merge
  request pipeline again, which re-runs the check and the labels on the
  archive commit; there is no approval click.
- Merge request pipelines from a fork run in the fork without the parent
  project's variables, so the fork branch of `spec:archive` prints the local
  commands and fails. **Do not enable running fork pipelines in the parent
  project to work around this**: GitLab then runs the CI configuration from
  the fork's branch with the parent's variables, including the archive
  token, which is the takeover this design exists to prevent. A fork's
  changes are archived by its author locally, or by a member after the
  branch is in this project.
- Enforcement is the project setting **Pipelines must succeed**; a failing
  `spec:check` blocks the merge.

## Maintainer actions

- Create the labels; create the token variables (`SPEC_ARCHIVE_TOKEN`
  required for the bot; `SPEC_LABELS_TOKEN` and `SPEC_NOTE_TOKEN` optional,
  `api` scope) as masked, unprotected variables.
- Enable "Pipelines must succeed" once `spec:check` has run.
- Record the observed behavior of the first archive run in the checks
  knowledge.

## Verification after installing

- The fragment parses; `grep -n '{{' .gitlab-ci.yml <fragment>` returns
  nothing.
- A test merge request: `spec:check` runs on each push; the manual jobs
  print the documents and the table; with the label applied and a pipeline
  run, `spec:archive` commits and pushes, and the next pipeline's
  `spec:check` passes.
