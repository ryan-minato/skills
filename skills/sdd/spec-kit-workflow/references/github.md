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

One rule carries this, and it is structural rather than a promise: **no
object authored by the request ever reaches a privileged runner.** A
privileged job (`pull_request_target`, `issue_comment`) checks out the
base and reads the head through `spec_kit_features.py snapshot`, which
pulls the file list and the documents from the REST API. The head's bytes
are parsed and never executed, so a later edit cannot turn a `git
checkout` typo into a takeover: the head is not in the runner's git store
to check out. Never add a checkout of the head, a `git fetch` of it, an
install from it, or a `run:` of its files.

- `spec / command` runs the default-branch workflow file and checkout, so
  every script comes from the base. Comment text reaches the shell only
  through `env`, with globbing off, and only as arguments to the script.
- `spec / labels` applies only labels that match the literal taxonomy
  written in the workflow, so a tampered script cannot make it apply an
  arbitrary one.
- The snapshot fetches only the documents the commands read and caps
  what one request can make a runner read: files touched, bytes per file,
  bytes in total, and API requests (the platform token's REST budget is
  shared by every workflow of the repository). Over a cap, or on a
  truncated tree, the job fails loudly instead of labeling on partial data.
- Request-authored names, paths, and task text reach a bot comment only
  inside code spans, and the echoed command loses its backticks, so a
  comment cannot carry a link or a mention under the bot's name.
- Commands admit only a collaborator association; the bot's own comments start
  no workflow run.

## Verification after installing

- Every workflow parses; actions are pinned by commit; `permissions: {}`
  at the top of each.
- No privileged workflow fetches or checks out the head: `grep -n 'git .*fetch\|checkout'`
  over the two files finds only the base checkouts.
- `snapshot --repo <o/r> --pr <n>` on a real pull request exits 0, and
  `status --snapshot` on its output matches `status --base ... --head ...`
  run against a local clone.
- `python3 scripts/spec_kit_features.py --help` exits 0; `check --draft`
  on a branch whose touched feature has an open task exits 0 with a
  warning, `check` exits 1; a touched feature without `plan.md` fails
  either way.
- A test pull request: `/spec status` gets a reply; the progress label
  follows the task list.
