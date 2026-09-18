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
| Status labels | `assets/github/workflow-spec-labels.yml` | `.github/workflows/spec-labels.yml` | job `spec / labels` |
| Labels | `assets/github/labels-spec.json` | rows in the project's label file | keep the project's own extra fields (an `applied_by` register, say); sync to the remote with the project's label tool |

Placeholders: `{{PINNED_SHA}}` (the commit SHA of the current release of
each action — read it from the action's releases page, and keep the
`# vX.Y.Z` comment the project's other workflows use), `{{NODE_LTS}}` and
`{{INSTALL_COMMAND}}` (the project's command that installs the pinned
OpenSpec CLI, so the version pin stays in one place).

## What the project's contract records

The archive executor — a person on the request's branch, never a job —
the job names, the label names, and the rule that the status labels are
workflow-owned. The project's checks knowledge lists the two jobs with
what a healthy run looks like, and the maintainer actions below.

## Maintainer actions

- Sync the five labels to the remote once (a dry run first, then apply).
- Record the observed behavior of the first live run (the comment
  permission, the applied labels) in the checks knowledge.

## What the installed jobs may do

Both privileged jobs read and comment with the platform token
(`GITHUB_TOKEN`) as `github-actions[bot]`. Neither writes to the
repository: `contents` stays at `read` everywhere, so nothing installed
here can push, and archiving stays a person's command. Consequences,
verified against GitHub's documentation on 2026-09-17:

- `issue_comment` events the token causes create no workflow run, so a
  reply the job posts never re-invokes it.
- No secret is stored or rotated. Nothing here needs one: an App or
  personal token buys only the ability to push, which no job does.
- A fork's own `pull_request` run gets a read-only token and no secrets,
  and the setting that would grant it write access exists for private
  repositories only — which is why labelling and replying on an external
  contribution run under privileged triggers at all.

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
- No installed job runs the CLI, installs anything, or needs the head's
  working tree, so neither ever checks the head out. Should a later change
  make one genuinely need it, guard the checkout with a literal
  `github.event.pull_request.head.repo.full_name == github.repository` in
  the step's own `if:` rather than behind an `env` variable: the guard is
  then visible at the step it guards, and code scanning recognizes it.
- The snapshot fetches only the documents the commands read and caps
  what one request can make a runner read: files touched, bytes per file,
  bytes in total, and API requests (the platform token's REST budget is
  shared by every workflow of the repository). Over a cap, or on a
  truncated tree, the job fails loudly instead of labeling on partial data.
- Request-authored names, paths, and task text reach a bot comment only
  inside code spans, and the echoed command loses its backticks, so a
  comment cannot carry a link or a mention under the bot's name.
- Invocation gates: the label job runs on the request's own events and
  applies only what the taxonomy allows; the commands admit only a
  collaborator association; the bot's own comments start nothing.

## Ready-state rules

A draft may carry unarchived related changes (the check warns). A ready
request needs every related change archived (the check fails otherwise),
the `Spec:` line (pointing at the change directory or its archive
directory), and the phase marker the project's template uses.

The request is marked ready before the archive, so `spec / check` is red
for the whole deliberation on the finished implementation, and the
archive commit is what turns it green. That red is the merge block by
design: nobody has to remember to hold the request, and nothing has to
be configured to stop it.

## Verification after installing

- Every workflow parses; actions are pinned by commit; `permissions: {}`
  at the top of each; no job grants `contents: write`; `GH_TOKEN` appears
  only in the `env` of the steps that call the API, never at the job level.
- No workflow pushes and none reaches the head:
  `grep -n 'git push\|git .*fetch\|checkout'` over the files finds only
  the base checkouts.
- `python3 scripts/spec_changes.py --help` exits 0; `check --draft` on a
  branch with an unarchived change exits 0 with a warning, `check` exits 1.
- `snapshot --repo <o/r> --pr <n>` on a real pull request exits 0, and
  `status --snapshot` on its output matches `status --base ... --head ...`
  run against a local clone.
- A test pull request: `/spec status` gets a reply, and the status labels
  match the change's task list.
