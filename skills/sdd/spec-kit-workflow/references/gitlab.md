# Installing the automation on GitLab

Read when installing or changing the Spec-Kit request automation in a
GitLab project. The assets under `assets/gitlab/` are raw shapes: every
`{{PLACEHOLDER}}` is replaced before the file lands.

## What lands where

| Piece | From | To | Notes |
|---|---|---|---|
| Script | `scripts/spec_kit_features.py` | the project's `scripts/spec_kit_features.py` | byte-identical copy |
| Jobs | `assets/gitlab/ci-spec-jobs.yml` | included from `.gitlab-ci.yml` (or pasted), stage `spec` | `spec:check`, `spec:show`, `spec:status`, `spec:labels` |
| Labels | `assets/gitlab/labels-spec.json` | the project's label file, synced with its label tool | GitLab colors carry `#` |

Placeholder: `{{PYTHON_IMAGE}}`. No archive job: Spec-Kit has no archive
operation; `spec:check` enforces the ticked task list once the merge
request is no longer a draft.

## What GitLab offers, and does not

Verified against GitLab's documentation on 2026-09-17: no pipeline source
exists for a merge request note or a label change, so `spec:show` and
`spec:status` are manual jobs (arguments from the manual job's variables
form: `SPEC_FEATURE`, `SPEC_DOC`) whose output goes to the job log and,
with `SPEC_NOTE_TOKEN` (`api` scope, masked), to a merge request note;
`spec:labels` applies the progress label with `SPEC_LABELS_TOKEN` or only
prints the plan; merge request pipelines from a fork run in the fork
without the parent's variables, so notes and labels work for a fork only
when a parent-project member runs the pipeline in the parent context;
enforcement is the project setting **Pipelines must succeed**.

## Maintainer actions

- Create the labels; create the optional token variables as masked,
  unprotected variables (merge request pipelines run on unprotected
  branches, so any job on any branch can read them — scope narrowly,
  rotate, record by name only).
- Enable "Pipelines must succeed" once `spec:check` has run.

## Verification after installing

- The fragment parses; `grep -n '{{' .gitlab-ci.yml <fragment>` returns
  nothing.
- A test merge request: `spec:check` runs on each push and fails on a
  touched feature without a plan; the manual jobs print the documents and
  the table.
