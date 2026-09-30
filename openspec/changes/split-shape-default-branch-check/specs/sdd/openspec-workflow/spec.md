## ADDED Requirements

### Requirement: Behavior: The default-branch check follows the recorded request shape
The agent SHALL configure the push branch of the produced check job with the change request shape the contract records, as its request branch already is — the push branch's `check --all` on GitHub, and both branches of `spec:check` on GitLab — so that no `check --all` falls back to the script's default; under the combined shape a push to the default branch SHALL fail on any change outside the archive directory, and under the split shape it SHALL pass with a warning naming each such change, whether or not a task of it is ticked, because refusing a request that completed a task is the request check's work; and the agent SHALL say the default branch never holds an unarchived change only for the combined shape, telling a split-shape project that its default branch holds each approved record until an implementation request archives it, so the warning is expected and no reason to archive.

#### Scenario: Split shape, push after a specification request
- **WHEN** the agent installs the automation in a GitHub repository whose contract records the split shape, and the produced job's push branch then runs on a default branch holding an approved change with no ticked task
- **THEN** every `check` command of the produced job carries `--shape split`, and the push branch exits 0 with a warning naming the change

#### Scenario: Split shape on GitLab
- **WHEN** the agent installs the automation in a GitLab project whose contract records the split shape, and the produced `spec:check` job runs on a default-branch pipeline holding an approved change with no ticked task
- **THEN** both branches of `spec:check` carry `--shape split`, and the job passes with a warning naming the change

#### Scenario: Split shape, a record with a ticked task
- **WHEN** the default branch of a split-shape project holds an unarchived change with a ticked task and the produced push check runs
- **THEN** it passes with a warning naming the change, because refusing a request that completed a task is the request check's work

#### Scenario: Combined shape, push
- **WHEN** the agent installs the automation in a project whose contract records the combined shape, and the produced push check runs on a default branch holding a change outside the archive directory
- **THEN** the push branch carries `--shape combined`, and the check fails and names the change

#### Scenario: Warning on a split project's default branch
- **WHEN** a user of a split-shape project with the automation installed says `spec / check` on the default branch warns that a change whose specification request merged is unarchived, and asks whether to archive it now to clear the warning
- **THEN** the agent says the warning is expected because the record awaits its implementation request, which archives it once its deliberation closes, and archives nothing

## MODIFIED Requirements

### Requirement: Behavior: The automation is installed per platform from the skill's assets
When asked to install the automation on GitHub, the agent SHALL produce from its assets a check job that joins the base's existing checks workflow and its gate (strict validation, and an unarchived related change as a warning on a draft and a failure when ready, so the merge stays shut for the whole deliberation; on a push to the default branch, any change outside the archive directory fails under the combined shape and is a warning naming it under the split shape), SHALL configure that job with the change request shape the contract records so that under the combined shape no request merges with an unarchived change while under the split shape the one request that implements nothing — the specification request — is admitted and any request that completed a task is not, the comment-command workflow, and the status-label workflow, SHALL add the five `spec/*` labels to the project's label file, SHALL deposit the management script `assets/spec_changes.py` as the project's own `scripts/spec_changes.py`, the path the workflows call — written for the workflows and owned by the project, with nothing requiring it to match the bundled script of the same name — and SHALL install no job that archives, commits, or pushes. The agent SHALL keep every workflow fork-safe by one structural rule — no object authored by the request reaches a privileged runner: scripts run from the base ref, no job checks the head out or fetches it, and the head is read through the management script's API snapshot; SHALL grant each job only the scopes its own reads and writes need, remembering that a permission left out of the block is none; SHALL restrict the comment command to the collaborator associations, so nobody without write access can start a privileged run; SHALL explain that a fork's unprivileged run holds a read-only token and no secrets and that the setting which would grant write exists for private repositories only, which is why labelling and replying need a privileged trigger at all; SHALL let no produced step hide an unexpected failure behind a fallback, and post a comment command's output as the reply even when the script fails — a deliberate deferral — failing the run afterwards for every failure except a bad argument; and SHALL record the maintainer action of syncing the labels once. On GitLab the agent SHALL produce the jobs fragment (check, manual show and status, labels), whose jobs keep a failing script's output and still fail, and state its limitations (no pipeline on a note or label change; manual jobs take no chat arguments; fork pipelines run in the fork) and SHALL refuse running a fork's pipeline in the parent project as a workaround, because that runs the fork's configuration with the parent's token.

#### Scenario: GitHub install
- **WHEN** the user asks to install the automation in a GitHub repository whose base has a checks workflow with a gate
- **THEN** the agent adds the check job to that workflow and its gate's dependencies, adds the two privileged workflows and the labels, deposits the management script from the skill's assets as the project's own `scripts/spec_changes.py`, names the label sync, and installs nothing that pushes

#### Scenario: Privileged job reads the head
- **WHEN** a produced workflow that runs with a writable token needs the head's documents
- **THEN** it checks out the base and reads them through the management script's API snapshot, and no step of any produced workflow fetches or checks the head out

#### Scenario: Command invoked by the request's author
- **WHEN** a comment command is posted by someone who is not a collaborator, including the request's own author
- **THEN** the produced workflow does not run

#### Scenario: Command answered when the script fails
- **WHEN** the management script exits non-zero while answering a comment command
- **THEN** the produced workflow still posts the script's output as the reply, and the run fails unless the script exited 2 for a bad argument such as an unknown change name

#### Scenario: Label plan checked before it is applied
- **WHEN** the label workflow applies the plan the script produced
- **THEN** it refuses and fails on any name outside the literal label taxonomy written in the workflow

#### Scenario: Asked for an archive bot
- **WHEN** the user asks for a label or job that archives the request's changes automatically
- **THEN** the agent declines, says the executor is a person because no platform token can push to a fork and the benefit does not pay for a job that holds write access to contents, and points at the archive command

#### Scenario: Ready with an unarchived change
- **WHEN** a ready pull request still carries an unarchived related change
- **THEN** the produced check fails and names the change, and the merge stays blocked until the record is archived

#### Scenario: Combined shape, nothing implemented
- **WHEN** a ready request on a combined-shape project holds an unarchived related change whose task list has no completed task
- **THEN** the produced check fails it, because a request that implemented nothing and did not freeze its record is unfinished rather than exempt

#### Scenario: Split shape, the specification request
- **WHEN** a ready request on a split-shape project carries the record alone, with no completed task, and the same project's later request completes one
- **THEN** the produced check admits the first and fails the second until its record is archived

#### Scenario: Fork pipeline in the parent project
- **WHEN** the user asks how to make the automation work for a fork's merge request on GitLab
- **THEN** the agent refuses the parent-project pipeline as a workaround and names the local commands instead

#### Scenario: GitLab install
- **WHEN** the user asks to install the automation in a GitLab project
- **THEN** the agent produces the jobs fragment and states that labels take effect on the next pipeline and that manual jobs replace comment commands

### Requirement: Script: spec_changes.py
The bundled script SHALL resolve a request's related changes from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer a `snapshot` subcommand that builds that document from the platform's REST API, listing the request's touched files and the related changes' documents without fetching or checking out the head, capping the files, bytes, and API calls it will read and refusing to report on a partial read of any kind — a cap reached, a truncated tree, or a document that is not decodable as text; a snapshot built for a different changes directory SHALL be refused rather than silently matched against none; `archive` SHALL accept only the git source, because it edits the working tree; SHALL offer `related`, `status`, `show`, `check`, `archive`, `labels`, and `snapshot` subcommands; `check` SHALL run the strict validator and fail on an unarchived related change (a warning with `--draft`; under `--shape split`, a warning for a related change with no ticked task) or, with `--all`, on any change outside the archive directory under the combined shape, the default, reporting each such change as a warning under `--shape split`; `archive` SHALL refuse to run while any path the tool writes is uncommitted, SHALL archive every related change whose tasks are all ticked through the tool's archive command (with the spec-less flag for a marked change), refuse all of them when any has an open task — an unchecked box with no text is an open task, validate strictly afterwards, and change nothing on a repeated run; `show` SHALL print a change's documents inside a fence, truncate at `--max-chars` with a link, and distinguish a document that was never written from one the source does not hold; `labels` SHALL compute the desired archive-axis and progress-axis labels and the additions and removals against the current set, and SHALL report the managed label names on request so a project's label file can be checked against them; every subcommand SHALL support `--help`, exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming the seven subcommands and exits 0

#### Scenario: Partial read refused
- **WHEN** the `snapshot` command reaches a cap, meets a truncated tree, or finds a document it cannot decode as text
- **THEN** it exits 1 naming the path and reports nothing, so no command runs on data it does not hold

#### Scenario: Snapshot built for another changes directory
- **WHEN** a command reads a snapshot whose recorded changes directory is not the one the command was given
- **THEN** it exits 2 and says to rebuild the snapshot, instead of matching no change and reporting an empty set

#### Scenario: Document never written
- **WHEN** `show` reaches a change that has no design of its own
- **THEN** it says the document was never written rather than that the source does not hold it

#### Scenario: Snapshot source matches the git source
- **WHEN** `snapshot` runs against a pull request and `status`, `show`, and `labels` run against the resulting document
- **THEN** their output is identical to the same subcommands run with `--base` and `--head` over a clone of that pull request

#### Scenario: Archive refuses a snapshot
- **WHEN** `archive` is invoked with `--snapshot`
- **THEN** it exits 2 and says the archive edits the working tree and needs the git source

#### Scenario: Representative run
- **WHEN** a related change with every task ticked exists at the checked-out head and `archive --base <base> --head HEAD` runs
- **THEN** the change moves under the archive directory, the main specs carry its delta, strict validation passes, and `check --base <base> --head HEAD` then exits 0

#### Scenario: Repeated run
- **WHEN** the identical `archive` command runs a second time
- **THEN** nothing changes and it exits 0

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option

#### Scenario: Open task refused
- **WHEN** one related change has an open task and another is complete
- **THEN** `archive` exits 1, names the open task, and archives neither

#### Scenario: Spec-less change
- **WHEN** a related change marked spec-less has every task ticked
- **THEN** `archive` moves it under the archive directory without touching the main specs

#### Scenario: Draft warning
- **WHEN** `check --draft` runs with an unarchived related change
- **THEN** it prints a warning naming the change and exits 0

#### Scenario: No related change
- **WHEN** the diff touches nothing under the changes directory
- **THEN** `labels` reports an empty desired set and removals for every managed label present

#### Scenario: Specification request under split
- **WHEN** `check --base <base> --head HEAD --shape split` runs on a diff touching an unarchived change with no ticked task, and runs again after one of its tasks is ticked
- **THEN** the first run exits 0 with a warning naming the change, and the second exits 1 naming it

#### Scenario: Integration branch under each shape
- **WHEN** the working tree holds one change outside the archive directory and `check --all` runs, then `check --all --shape split`
- **THEN** the first exits 1 naming the change, the second exits 0 with a warning naming it, and `check --help` describes `--all` under both shapes

### Requirement: Script: assets/spec_changes.py
The management script the automation deposits SHALL resolve a request's related changes from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer the `snapshot`, `check`, `show`, `status`, and `labels` subcommands and none that edits the working tree; `snapshot` SHALL build that document from the platform's REST API without fetching or checking out the head, SHALL check every response it relies on — its shape, the truncation flag, the encoding, and text decoding — and SHALL fail naming the path or endpoint, writing no document, on a cap reached or on any partial or unexpected read; a snapshot built for a different changes directory SHALL be refused; `check` SHALL run the strict validator and fail on an unarchived related change (a warning with `--draft`, the specification request admitted under `--shape split`) or, with `--all`, on any change outside the archive directory under the combined shape, the default, reporting each such change as a warning under `--shape split`; `show` SHALL print a change's documents inside a fence and truncate at `--max-chars` with a link; `show` and `status` SHALL answer a change name the request does not touch with the related changes' names, every name rendered as code; `labels` SHALL compute the desired archive-axis and progress-axis labels and the additions and removals against the current set, and SHALL report the managed label names with `--taxonomy`; the script SHALL exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments, an unknown change name included, and SHALL let an unexpected error end with its traceback.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming the five subcommands and exits 0

#### Scenario: Representative run
- **WHEN** a related change with every task ticked is archived at the head and `check --base <base> --head HEAD` runs
- **THEN** it exits 0, and `status` with the same refs reports the change as done

#### Scenario: Repeated run
- **WHEN** the identical `check` and `status` commands run a second time
- **THEN** the output is identical and nothing in the tree changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option

#### Scenario: Unknown change named
- **WHEN** `show` is given a change name the request does not touch
- **THEN** it exits 2 and lists the related changes, and every change name in its output, the given one included, is rendered as code

#### Scenario: Partial read refused
- **WHEN** `snapshot` reaches a cap, meets a truncated tree, or finds a document it cannot decode as text
- **THEN** it exits 1 naming the path and writes no document

#### Scenario: Unexpected API response
- **WHEN** a response `snapshot` relies on is not the documented shape, or a tree that a listing named answers 404
- **THEN** `snapshot` fails naming the endpoint and writes no document, instead of reading the response as empty

#### Scenario: Snapshot source matches the git source
- **WHEN** `snapshot` runs against a pull request and `status`, `show`, and `labels` run against the resulting document
- **THEN** their output is identical to the same subcommands run with `--base` and `--head` over a clone of that pull request

#### Scenario: Integration branch under each shape
- **WHEN** the working tree holds one change outside the archive directory and `check --all` runs, then `check --all --shape split`
- **THEN** the first exits 1 naming the change, the second exits 0 with a warning naming it, and `check --help` describes `--all` under both shapes
