## MODIFIED Requirements

### Requirement: Behavior: The automation is installed per platform from the skill's assets
When asked to install the automation on GitHub, the agent SHALL produce from its assets a check job that joins the base's existing checks workflow and its gate (required files present for every touched feature; an open task as a warning on a draft and a failure when ready), SHALL configure that job with the change request shape the contract records so that under the combined shape no request merges unfinished while under the split shape the one request that implements nothing — the specification request — is admitted and any request that completed a task is not, the comment-command workflow (`/spec show` and `/spec status` over touched features), and the progress-label workflow, SHALL add the three progress labels to the project's label file, SHALL deposit the management script `assets/spec_kit_request.py` as the project's own `scripts/spec_kit_request.py` — written for the workflows, owned by the project, and never a copy of the bundled `scripts/spec_kit_features.py` — SHALL grant each job only the scopes its own reads and writes need, remembering that a permission left out of the block is none, SHALL keep every workflow fork-safe by one structural rule — no object authored by the request reaches a privileged runner: scripts run from the base ref, no job checks the head out or fetches it, and the head is read through the management script's API snapshot, SHALL restrict the comment command to the collaborator associations so nobody without write access can start a privileged run, SHALL explain that a fork's unprivileged run holds a read-only token and no secrets and that the setting which would grant write exists for private repositories only, which is why labelling and replying need a privileged trigger at all, SHALL run every produced shell step under bash with `errexit` and `pipefail`, and post a comment command's output as the reply even when the script fails, failing the run for every failure except a bad argument, and SHALL say that no job archives or pushes, both because the kit has no archive operation and because the executor of a freeze is a person; on GitLab the agent SHALL produce the jobs fragment, whose jobs keep a failing script's output and still fail, state its limitations, and refuse running a fork's pipeline in the parent project as a workaround, because that runs the fork's configuration with the parent's token.

#### Scenario: GitHub install
- **WHEN** the user asks to install the automation in a GitHub repository whose base has a checks workflow with a gate
- **THEN** the agent adds the check job to that workflow and its gate's dependencies, adds the two privileged workflows and the labels, deposits the management script as the project's own `scripts/spec_kit_request.py` with no file byte-identical to the bundled script, and says nothing it installs archives or pushes

#### Scenario: Combined shape, nothing implemented
- **WHEN** a ready request on a combined-shape project touches a feature whose task list has no completed task
- **THEN** the produced check fails it, because a request that implemented nothing is unfinished rather than exempt

#### Scenario: Split shape, the specification request
- **WHEN** a ready request on a split-shape project carries the specification and the plan with no completed task
- **THEN** the produced check admits it, and fails a later request of the same project that completed one and left the rest open

#### Scenario: Privileged job reads the head
- **WHEN** a produced workflow that runs with a writable token needs a touched feature's documents
- **THEN** it checks out the base and reads them through the management script's API snapshot, and no step of any produced workflow fetches or checks the head out

#### Scenario: Ready with an open task
- **WHEN** a ready pull request touches a feature whose task list has an open item
- **THEN** the produced check fails and names the feature and the task

#### Scenario: Command invoked by the request's author
- **WHEN** a comment command is posted by someone who is not a collaborator, including the request's own author
- **THEN** the produced workflow does not run

#### Scenario: Command answered when the script fails
- **WHEN** the management script exits non-zero while answering a comment command
- **THEN** the produced workflow still posts the script's output as the reply, and the run fails unless the script exited 2 for a bad argument such as an unknown feature name

## ADDED Requirements

### Requirement: Script: spec_kit_request.py
The management script the automation deposits SHALL resolve a request's touched features (numbered directories under the kit's specs directory whose files the request touches) from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer the `snapshot`, `check`, `show`, `status`, and `labels` subcommands and none that edits the working tree; `snapshot` SHALL build that document from the platform's REST API without fetching or checking out the head, SHALL check every response it relies on — its shape, the truncation flag, the encoding, and text decoding — and SHALL fail naming the path or endpoint, writing no document, on a cap reached or on any partial or unexpected read; a snapshot built for a different specs directory SHALL be refused; `check` SHALL fail when a touched feature lacks its specification or plan, or when a touched feature has an open task and the request is ready (a warning with `--draft`, the specification request admitted under `--shape split`), an unchecked box with no text counting as an open task; `show` SHALL print a feature's documents inside a fence and truncate at `--max-chars` with a link; `show` and `status` SHALL answer a feature name the request does not touch with the touched features' names, every name rendered as code; `labels` SHALL compute the desired progress label and the additions and removals against the current set, and SHALL report the managed label names with `--taxonomy`; the script SHALL exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments, an unknown feature name included, and SHALL let an unexpected error end with its traceback.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming the five subcommands and exits 0

#### Scenario: Representative run
- **WHEN** a touched feature has its specification, plan, and a fully ticked task list and `check --base <base> --head <head>` runs
- **THEN** it exits 0, and `status` with the same refs reports the feature as done

#### Scenario: Repeated run
- **WHEN** the identical `check` and `status` commands run a second time
- **THEN** the output is identical and nothing in the tree changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option

#### Scenario: Unknown feature named
- **WHEN** `show` is given a feature name the request does not touch
- **THEN** it exits 2 and lists the touched features, and every feature name in its output, the given one included, is rendered as code

#### Scenario: Missing plan
- **WHEN** a touched feature has a specification but no plan
- **THEN** `check` exits 1 and names the feature and the missing file

#### Scenario: Partial read refused
- **WHEN** `snapshot` reaches a cap, meets a truncated tree, or finds a document it cannot decode as text
- **THEN** it exits 1 naming the path and writes no document

#### Scenario: Unexpected API response
- **WHEN** a response `snapshot` relies on is not the documented shape, or a tree that a listing named answers 404
- **THEN** `snapshot` fails at once naming the endpoint and writes no document, instead of reading the response as empty

#### Scenario: Snapshot source matches the git source
- **WHEN** `snapshot` runs against a pull request and `status`, `show`, and `labels` run against the resulting document
- **THEN** their output is identical to the same subcommands run with `--base` and `--head` over a clone of that pull request
