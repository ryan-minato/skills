# Installing the automation on GitLab

Read when installing or changing the OpenSpec request automation in a
GitLab project. The assets under `assets/gitlab/` are raw shapes: every
`{{PLACEHOLDER}}` is replaced before the file lands.

## What lands where

| Piece | From | To | Notes |
|---|---|---|---|
| Script | `scripts/spec_changes.py` | the project's `scripts/spec_changes.py` | byte-identical copy |
| Jobs | `assets/gitlab/ci-spec-jobs.yml` | included from `.gitlab-ci.yml` (or pasted), stage `spec` | `spec:check`, `spec:show`, `spec:status`, `spec:labels` |
| Labels | `assets/gitlab/labels-spec.json` | the project's label file, synced with its label tool | GitLab colors carry `#`; on Premium the two axes may become scoped labels for built-in exclusivity |

Placeholders: `{{NODE_IMAGE}}`, `{{INSTALL_COMMAND}}`.

## What GitLab offers, and does not

Verified against GitLab's documentation on 2026-09-17:

- No pipeline source exists for a merge request note or a label change.
  Merge request pipelines run when the request is created, when its source
  branch receives a push, from the **Run pipeline** button, or from the
  `/run_pipeline` quick action (GitLab 18.7 and later). A label a job
  applies therefore appears only after the next pipeline runs.
- Quick actions carry no arguments a job could read, so `spec:show` and
  `spec:status` are manual jobs whose arguments come from the manual job's
  variables form (`SPEC_CHANGE`, `SPEC_DOC`); their output goes to the job
  log and, with `SPEC_NOTE_TOKEN`, to a merge request note.
- No job here pushes, so no job needs a token with `write_repository`.
  The optional `SPEC_LABELS_TOKEN` and `SPEC_NOTE_TOKEN` carry `api` only.
  Merge request pipelines run on unprotected source branches, so such a
  variable must be unprotected and any job on any branch of the project
  can read it — scope it narrowly, rotate it, and record it in the
  settings knowledge by name only.
- Merge request pipelines from a fork run in the fork without the parent
  project's variables, so the label and note jobs simply print their plan
  there. **Do not enable running fork pipelines in the parent project to
  work around this**: GitLab then runs the CI configuration from the
  fork's branch with the parent's variables, which is the takeover this
  design exists to prevent. Archiving is a person's command on the branch
  in either case — its author locally, or a member after pulling it.
- Enforcement is the project setting **Pipelines must succeed**; a failing
  `spec:check` blocks the merge.

## Maintainer actions

- Create the labels; create the optional token variables
  (`SPEC_LABELS_TOKEN` and `SPEC_NOTE_TOKEN`, `api` scope) as masked,
  unprotected variables. Without them the jobs print their plan and change
  nothing.
- Enable "Pipelines must succeed" once `spec:check` has run.
- Record the observed behavior of the first live run in the checks
  knowledge.

## Verification after installing

- The fragment parses; `grep -n '{{' .gitlab-ci.yml <fragment>` returns
  nothing.
- A test merge request: `spec:check` runs on each push; the manual jobs
  print the documents and the table; `spec:labels` plans the two axes from
  the change's task list.
