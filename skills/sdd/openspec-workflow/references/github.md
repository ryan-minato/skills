# Installing the automation on GitHub

Read when installing or changing the OpenSpec request automation in a
GitHub repository. The assets under `assets/github/` are raw shapes: every
`{{PLACEHOLDER}}` is replaced, and `grep -rn '{{[A-Z]' .github/` returns
nothing when the install is done (Actions expressions `${{ … }}` are not
placeholders).

## What lands where

| Piece | From | To | Notes |
|---|---|---|---|
| Script | `scripts/spec_changes.py` | the project's `scripts/spec_changes.py` | byte-identical copy; the workflows call it by that path |
| Check | `assets/github/job-spec-check.yml` | a job in the project's existing checks workflow | add `spec-check` to the aggregator gate's `needs:`; set `on.pull_request.types` to `[opened, synchronize, reopened, ready_for_review, converted_to_draft]` so the verdict follows the draft state; no new required check — the gate is already required |
| Comment commands | `assets/github/workflow-spec-command.yml` | `.github/workflows/spec-command.yml` | job `spec / command` |
| Archive bot | `assets/github/workflow-spec-archive.yml` | `.github/workflows/spec-archive.yml` | job `spec / archive` |
| Status labels | `assets/github/workflow-spec-labels.yml` | `.github/workflows/spec-labels.yml` | job `spec / labels` |
| Labels | `assets/github/labels-spec.json` | rows in the project's label file | keep the project's own extra fields (an `applied_by` register, say); sync to the remote with the project's label tool |

Placeholders: `{{PINNED_SHA}}` (the commit SHA of the current release of
each action — read it from the action's releases page, and keep the
`# vX.Y.Z` comment the project's other workflows use), `{{NODE_LTS}}`,
`{{INSTALL_COMMAND}}` (the project's command that installs the pinned
OpenSpec CLI, so the version pin stays in one place),
`{{VALIDATE_COMMAND}}` (the project's strict-validation command, quoted to
the fork author), `{{ARCHIVE_COMMIT_PREFIX}}` and
`{{ARCHIVE_COMMIT_SUBJECT}}` (per the project's commit convention, for
example `docs:` and `docs: archive the <name> change`).

## What the project's contract records

The executor (`the spec/archive label`, with by-hand archiving still
allowed), the job names, the label names, and the rule that the status
labels are workflow-owned. The project's checks knowledge lists the three
jobs with what a healthy run looks like, and the maintainer actions below.

## Maintainer actions

- Sync the six labels to the remote once (a dry run first, then apply).
- After every bot push: click **Approve workflows to run** in the merge
  box. The bot's summary comment asks for it each time.
- Record the observed behavior of the first live run (the approval banner,
  the label removal, the comment permission) in the checks knowledge.

## Bot identity

The bot pushes and comments with the platform token (`GITHUB_TOKEN`) as
`github-actions[bot]`. Consequences, verified against GitHub's
documentation on 2026-09-17:

- A push made with the token puts the resulting `pull_request` runs
  (`synchronize`) in an approval-required state; a user with write access
  starts them. Runs left waiting for more than 30 days are deleted.
- `labeled` events and `issue_comment` events the token causes create no
  workflow run, so removing the trigger label and posting the summary
  never recurse.
- No secret is stored or rotated. An App or personal token would start the
  runs automatically at the cost of a secret in every project and a user's
  identity on the archive commit; it is not the default.

## Fork safety

One rule carries this, and it is structural rather than a promise: **no
object authored by the request ever reaches a privileged runner.** A
privileged job (`pull_request_target`, `issue_comment`) checks out the
base and reads the head through `spec_changes.py snapshot`, which pulls
the file list and the documents from the REST API. The head's bytes are
parsed and never executed, so a later edit cannot turn a `git checkout`
typo into a takeover: the head is not in the runner's git store to check
out.

- `spec / command` runs the default-branch workflow file and checkout, so
  every script comes from the base. Comment text reaches the shell only
  through `env`, with globbing off, and only as arguments to the script.
- `spec / labels` never touches git beyond the base checkout. The label
  plan it applies is checked against the literal label taxonomy in the
  workflow before any API call, so a tampered script cannot make it apply
  an arbitrary label.
- `spec / archive` is the one job that needs the head's working tree,
  because it runs the CLI over it. It checks the head out only under a
  literal `github.event.pull_request.head.repo.full_name ==
  github.repository` condition on the step — such a branch lives in the
  base repository and its code already runs in the project's own CI. For a
  fork it reads the snapshot, posts the commands, and pushes nothing.
  Write the comparison in the `if:` itself rather than behind an `env`
  variable: the guard is then visible at the step it guards, and code
  scanning recognizes it.
- The snapshot fetches only the documents the commands read and caps
  what one request can make a runner read: files touched, bytes per file,
  bytes in total, and API requests (the platform token's REST budget is
  shared by every workflow of the repository). Over a cap, or on a
  truncated tree, the job fails loudly instead of labeling on partial data.
- Request-authored names, paths, and task text reach a bot comment only
  inside code spans, and the echoed command loses its backticks, so a
  comment cannot carry a link or a mention under the bot's name.
- Invocation gates: labeling needs triage access; commands need a
  collaborator or the author; the bot's own comments start nothing.

## Ready-state rules

A draft may carry unarchived related changes (the check warns). A ready
request needs every related change archived (the check fails otherwise),
the `Spec:` line (pointing at the change directory or its archive
directory), and the phase marker the project's template uses.

## Verification after installing

- Every workflow parses; actions are pinned by commit; `permissions: {}`
  at the top of each; the archive workflow's fork step contains no
  `git push`.
- No privileged workflow fetches or checks out the head: `grep -n 'git .*fetch\|checkout'`
  over the three files finds only the base checkouts and the
  same-repository checkout guarded by the literal `head.repo.full_name`
  condition.
- `python3 scripts/spec_changes.py --help` exits 0; `check --draft` on a
  branch with an unarchived change exits 0 with a warning, `check` exits 1.
- `snapshot --repo <o/r> --pr <n>` on a real pull request exits 0, and
  `status --snapshot` on its output matches `status --base ... --head ...`
  run against a local clone.
- A test pull request: `/spec status` gets a reply; applying
  `spec/archive` on a complete change produces the commit, the summary, and
  the removed label; the approval banner appears.
