# Installing the automation on GitHub

Read when installing or changing the Spec-Kit request automation in a
GitHub repository. The assets under `assets/github/` are raw shapes: every
`{{PLACEHOLDER}}` is replaced, and `grep -rn '{{[A-Z]' .github/` returns
nothing when the install is done (Actions expressions `${{ … }}` are not
placeholders).

## What lands where

| Piece | From | To | Notes |
|---|---|---|---|
| Script | `scripts/spec_kit_features.py` | the project's `scripts/spec_kit_features.py` | byte-identical copy; the workflows call it by that path |
| Check | `assets/github/job-spec-check.yml` | a job in the project's existing checks workflow | add `spec-check` to the aggregator gate's `needs:`; set `on.pull_request.types` to `[opened, synchronize, reopened, ready_for_review, converted_to_draft]`; no new required check — the gate is already required |
| Comment commands | `assets/github/workflow-spec-command.yml` | `.github/workflows/spec-command.yml` | job `spec / command` |
| Progress label | `assets/github/workflow-spec-labels.yml` | `.github/workflows/spec-labels.yml` | job `spec / labels` |
| Labels | `assets/github/labels-spec.json` | rows in the project's label file | keep the project's own extra fields; sync with the project's label tool |

No archive bot: Spec-Kit has no archive operation. Completion is every
task of every touched feature ticked, which the check enforces once the
pull request is ready. Placeholder: `{{PINNED_SHA}}` (the commit SHA of
the current release of the checkout action, with the `# vX.Y.Z` comment
the project's other workflows use).

## What the project's contract records

The completion rule (every task ticked before ready), the job names, the
label names, and the rule that the progress label is workflow-owned; when
the level is spec-anchored, the project's rule for updating the living
specification, which nothing in the kit enforces.

## Maintainer actions

- Sync the three labels to the remote once (a dry run first, then apply).
- Record the observed behavior of the first live run (the comment
  permission) in the checks knowledge.

## Fork safety

- `spec / command` runs the default-branch workflow file and checkout;
  the head is fetched by its exact SHA as git objects and read by the
  base's script. Comment text reaches the shell only through `env`,
  with globbing off, and only as arguments to the script.
- `spec / labels` uses `pull_request_target`, whose token is writable even
  for a fork; it is safe because the checkout is the base, the head is
  fetched by its SHA as objects, and nothing from the head executes. Never
  add a checkout of the head, an install from it, or a `run:` of its
  files; a code-scanning finding on the fetch is reviewed against these
  rules and dismissed by the maintainer, never satisfied by a checkout.
- Commands need a collaborator or the author; the bot's own comments start
  no workflow run.

## Verification after installing

- Every workflow parses; actions are pinned by commit; `permissions: {}`
  at the top of each.
- `python3 scripts/spec_kit_features.py --help` exits 0; `check --draft`
  on a branch whose touched feature has an open task exits 0 with a
  warning, `check` exits 1; a touched feature without `plan.md` fails
  either way.
- A test pull request: `/spec status` gets a reply; the progress label
  follows the task list.
