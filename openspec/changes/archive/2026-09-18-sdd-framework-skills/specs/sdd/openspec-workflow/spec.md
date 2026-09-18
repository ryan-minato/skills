## Purpose
Governs what an agent that loaded the `openspec-workflow` skill observably does when a project runs OpenSpec changes through pull or merge requests: the approval package, the in-request archive that freezes the record after the deliberation, the comment commands and status labels, and the automation it installs per platform.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project that runs OpenSpec asks how its changes pass approval and archiving through pull or merge requests, how to use or install the `/spec` comment commands and the status labels, or how to wire the OpenSpec check into CI, and SHALL not cause it to load for a tool command that creates or applies a single change, for choosing a spec tool, or for a project that runs another framework.

#### Scenario: Archive question
- **WHEN** the user says "our OpenSpec change is done; how does it get archived before this PR merges?"
- **THEN** the skill loads

#### Scenario: Automation install
- **WHEN** the user says "add the /spec comment commands and the status labels to this GitHub repository"
- **THEN** the skill loads

#### Scenario: Tool command request
- **WHEN** the user says "create a new OpenSpec change for the export feature"
- **THEN** the skill does not load

#### Scenario: Tool choice
- **WHEN** the user says "should we use OpenSpec or Spec-Kit for this library?"
- **THEN** the skill does not load

### Requirement: Behavior: The approval package is the proposal, the delta specs, and the design when warranted
The agent SHALL treat `proposal.md`, the delta specs, and `design.md` when the project's rule warrants one as the approval package the draft opens with (by default: more than one reasonable approach, or the change touches structure, interfaces, dependencies, or files outside the record; a wording change inside one section needs none), SHALL treat `tasks.md` as after-approval material even when the tool's propose step generated it, SHALL mark a change to the project's own harness, tooling, checks, workflows, or documents with `skip_specs: true` in its change configuration so it carries no delta spec, and SHALL run the strict validator after each artifact edit, before publishing the draft, before ready, and after archiving.

#### Scenario: Propose generated all four files
- **WHEN** the tool's propose step generated the proposal, the delta specs, the design, and the task list
- **THEN** the agent finishes the design, publishes the draft with the first three as the package, and marks the task list as after-approval

#### Scenario: Harness change
- **WHEN** the change edits the project's CI workflow
- **THEN** the agent sets the spec-less marker, writes a proposal, a design, and tasks, and creates no delta spec

### Requirement: Behavior: A change is archived inside the request once the deliberation closes
The agent SHALL mark the request ready so the finished implementation can be deliberated on, SHALL archive every change the request touches only after that deliberation closes, through the tool's archive command with its non-interactive flag (and its spec-less flag for a marked change), and SHALL treat that archive as the freeze the approval applies to; SHALL refuse to archive a change with an open task; on a request from a fork SHALL name a maintainer as the executor on the contributor's branch, because no job of the automation pushes anywhere; and SHALL never edit an archived record.

#### Scenario: After the deliberation
- **WHEN** the deliberation on the finished implementation has closed and every task is ticked
- **THEN** the agent runs the archive command non-interactively, runs the strict validator, and commits the archived record on the request's branch

#### Scenario: Asked to archive before the deliberation
- **WHEN** the implementation is finished and the user asks to archive and mark the request ready together
- **THEN** the agent marks it ready, archives nothing yet, and says the freeze follows the deliberation

#### Scenario: Open task
- **WHEN** the user asks to archive a change whose task list has an open item
- **THEN** the agent refuses, names the open task, and archives nothing

#### Scenario: Fork
- **WHEN** the request comes from a fork and its deliberation has closed
- **THEN** the agent says a maintainer archives on the contributor's branch and that nothing in the installed automation can push there

### Requirement: Behavior: Comment commands and status labels are read and used as the project installed them
The agent SHALL use `/spec show [<change>] [proposal|design|tasks|specs|all]` and `/spec status [<change>]` on a request to show a related change's documents and task progress, SHALL read the archive axis (`spec/archived`, `spec/unarchived`) and the progress axis (`spec/not-started`, `spec/in-progress`, `spec/done`) as facts the workflows maintain and never apply or remove them by hand, the archive axis being what tells a reader of the request list whether a ready request is still being deliberated on or already frozen, and SHALL define a request's related changes as the change directories whose files the request's diff touches, an archived directory counting under its change name.

#### Scenario: Progress asked
- **WHEN** the user asks how far the request's change is
- **THEN** the agent points at the progress label or posts `/spec status` and reads the reply

#### Scenario: Label edited by hand
- **WHEN** the user asks the agent to set `spec/done` on the request
- **THEN** the agent declines and says the labels workflow derives it from the task list

### Requirement: Behavior: The automation is installed per platform from the skill's assets
When asked to install the automation on GitHub, the agent SHALL produce from its assets a check job that joins the base's existing checks workflow and its gate (strict validation, and an unarchived related change as a warning on a draft and a failure when ready, so the merge stays shut for the whole deliberation; on a push to the default branch, any change outside the archive directory fails), SHALL configure that job with the change request shape the contract records so that under the combined shape no request merges with an unarchived change while under the split shape the one request that implements nothing — the specification request — is admitted and any request that completed a task is not, the comment-command workflow, and the status-label workflow, SHALL add the five `spec/*` labels to the project's label file, SHALL copy `scripts/spec_changes.py` into the project's scripts, and SHALL install no job that archives, commits, or pushes. The agent SHALL keep every workflow fork-safe by one structural rule — no object authored by the request reaches a privileged runner: scripts run from the base ref, no job checks the head out or fetches it, and the head is read through the script's API snapshot; SHALL grant each job only the scopes its own reads and writes need, remembering that a permission left out of the block is none; SHALL restrict the comment command to the collaborator associations, so nobody without write access can start a privileged run; SHALL explain that a fork's unprivileged run holds a read-only token and no secrets and that the setting which would grant write exists for private repositories only, which is why labelling and replying need a privileged trigger at all; and SHALL record the maintainer action of syncing the labels once. On GitLab the agent SHALL produce the jobs fragment (check, manual show and status, labels) and state its limitations (no pipeline on a note or label change; manual jobs take no chat arguments; fork pipelines run in the fork) and SHALL refuse running a fork's pipeline in the parent project as a workaround, because that runs the fork's configuration with the parent's token.

#### Scenario: GitHub install
- **WHEN** the user asks to install the automation in a GitHub repository whose base has a checks workflow with a gate
- **THEN** the agent adds the check job to that workflow and its gate's dependencies, adds the two privileged workflows and the labels, copies the script, names the label sync, and installs nothing that pushes

#### Scenario: Privileged job reads the head
- **WHEN** a produced workflow that runs with a writable token needs the head's documents
- **THEN** it checks out the base and reads them through the script's API snapshot, and no step of any produced workflow fetches or checks the head out

#### Scenario: Command invoked by the request's author
- **WHEN** a comment command is posted by someone who is not a collaborator, including the request's own author
- **THEN** the produced workflow does not run

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

### Requirement: Handoff: the methodology skill
The agent SHALL route questions about whether or at which level to adopt spec-driven development, what a good specification is, or the generic loop to the methodology skill of the `sdd` catalog through the installing skill, and when the user declines SHALL answer from the loop facts this skill restates (the package before the task list, the freeze before approval) without teaching the practice.

#### Scenario: Handoff offered
- **WHEN** the user asks whether the project should adopt spec-driven development at all
- **THEN** the agent offers the methodology skill through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the methodology skill
- **THEN** the agent answers the OpenSpec question at hand and says the practice-level question stays open

### Requirement: Handoff: the contract builder
The agent SHALL route setting or changing the project's rules — the approval mode, when a design is warranted, the archive executor, the request shape — to the spec workflow builder of the harness catalog through the installing skill, and when the user declines SHALL apply the contract as it stands or the defaults (the implementer archives once the deliberation on the finished implementation closes, and both gates close in conversation) and say so.

#### Scenario: Handoff offered
- **WHEN** the user wants the project to record a different archive executor or approval mode
- **THEN** the agent offers the spec workflow builder through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the builder
- **THEN** the agent installs the automation, applies the defaults for what the contract does not say, and names the contract update as remaining work

### Requirement: Script: spec_changes.py
The bundled script SHALL resolve a request's related changes from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer a `snapshot` subcommand that builds that document from the platform's REST API, listing the request's touched files and the related changes' documents without fetching or checking out the head, capping the files, bytes, and API calls it will read and refusing to report on a partial read of any kind — a cap reached, a truncated tree, or a document that is not decodable as text; a snapshot built for a different changes directory SHALL be refused rather than silently matched against none; `archive` SHALL accept only the git source, because it edits the working tree; SHALL offer `related`, `status`, `show`, `check`, `archive`, `labels`, and `snapshot` subcommands; `check` SHALL run the strict validator and fail on an unarchived related change (a warning with `--draft`) or, with `--all`, on any change outside the archive directory; `archive` SHALL refuse to run while any path the tool writes is uncommitted, SHALL archive every related change whose tasks are all ticked through the tool's archive command (with the spec-less flag for a marked change), refuse all of them when any has an open task — an unchecked box with no text is an open task, validate strictly afterwards, and change nothing on a repeated run; `show` SHALL print a change's documents inside a fence, truncate at `--max-chars` with a link, and distinguish a document that was never written from one the source does not hold; `labels` SHALL compute the desired archive-axis and progress-axis labels and the additions and removals against the current set, and SHALL report the managed label names on request so a project's label file can be checked against them; every subcommand SHALL support `--help`, exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments.

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
