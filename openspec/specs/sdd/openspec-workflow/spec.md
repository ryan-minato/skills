# sdd/openspec-workflow Specification

## Purpose
Governs what an agent that loaded the `openspec-workflow` skill observably does when a project runs OpenSpec changes through pull or merge requests: the approval package, the in-request archive by hand or by the label bot, the comment commands and status labels, and the automation it installs per platform.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project that runs OpenSpec asks how its changes pass approval and archiving through pull or merge requests, how to use or install the `/spec` comment commands, the archive label, and the status labels, or how to wire the OpenSpec check into CI, and SHALL not cause it to load for a tool command that creates or applies a single change, for choosing a spec tool, or for a project that runs another framework.

#### Scenario: Archive question
- **WHEN** the user says "our OpenSpec change is done; how does it get archived before this PR merges?"
- **THEN** the skill loads

#### Scenario: Automation install
- **WHEN** the user says "add the /spec comment commands and the archive label bot to this GitHub repository"
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

### Requirement: Behavior: A change is archived inside the request, by hand or by the label bot
The agent SHALL archive every change the request touches before it is marked ready, through the tool's archive command with its non-interactive flag (and its spec-less flag for a marked change) when the executor is by hand, or by applying the trigger label as an authorized remote write and waiting for the bot's commit when the contract names the label; SHALL refuse to archive a change with an open task; on a fork SHALL run the commands the bot posts and push them; and SHALL never edit an archived record.

#### Scenario: By hand
- **WHEN** the contract records the by-hand executor and every task is ticked
- **THEN** the agent runs the archive command non-interactively, runs the strict validator, and commits the archived record on the request's branch

#### Scenario: Open task
- **WHEN** the user asks to archive a change whose task list has an open item
- **THEN** the agent refuses, names the open task, and archives nothing

#### Scenario: Label bot
- **WHEN** the contract records the trigger label and the user authorizes applying it
- **THEN** the agent applies the label, waits for the bot's commit and summary comment, pulls the branch, and reminds the user that a write-access user must approve the pending workflow runs

### Requirement: Behavior: Comment commands and status labels are read and used as the project installed them
The agent SHALL use `/spec show [<change>] [proposal|design|tasks|specs|all]` and `/spec status [<change>]` on a request to show a related change's documents and task progress, SHALL read the archive axis (`spec/archived`, `spec/unarchived`) and the progress axis (`spec/not-started`, `spec/in-progress`, `spec/done`) as facts the workflows maintain and never apply or remove them by hand, and SHALL define a request's related changes as the change directories whose files the request's diff touches, an archived directory counting under its change name.

#### Scenario: Progress asked
- **WHEN** the user asks how far the request's change is
- **THEN** the agent points at the progress label or posts `/spec status` and reads the reply

#### Scenario: Label edited by hand
- **WHEN** the user asks the agent to set `spec/done` on the request
- **THEN** the agent declines and says the labels workflow derives it from the task list

### Requirement: Behavior: The automation is installed per platform from the skill's assets
When asked to install the automation on GitHub, the agent SHALL produce from its assets a check job that joins the base's existing checks workflow and its gate (strict validation, and an unarchived related change as a warning on a draft and a failure when ready; on a push to the default branch, any change outside the archive directory fails), the comment-command workflow, the label-triggered archive workflow (same-repository branch: archive, commit, push with the platform token, remove the label, summarize; fork: mention the author with the commands, push nothing, remove the label), and the status-label workflow, SHALL add the six `spec/*` labels to the project's label file, SHALL copy `scripts/spec_changes.py` into the project's scripts, SHALL keep every workflow fork-safe by one structural rule — no object authored by the request reaches a privileged runner: scripts run from the base ref, the head is read through the script's API snapshot rather than fetched into the runner's git store, and the only job that needs the head's working tree checks it out under a literal `head.repo.full_name == github.repository` comparison written in the step's own condition, SHALL grant each job only the scopes its own reads and writes need, remembering that a permission left out of the block is none, SHALL state that a push made with the platform token puts the resulting pull-request runs in an approval-required state that a write-access user starts, and SHALL record the maintainer actions (label sync; the approval click after each bot push); on GitLab the agent SHALL produce the jobs fragment (check, manual show and status, label-gated or manual archive, labels) and state its limitations (no pipeline on a note or label change, so the label is applied and a pipeline started; manual jobs take no chat arguments; a project token variable for the source-branch push; fork pipelines run in the fork) and SHALL refuse running a fork's pipeline in the parent project as a workaround, because that runs the fork's configuration with the parent's token.

#### Scenario: GitHub install
- **WHEN** the user asks to install the automation in a GitHub repository whose base has a checks workflow with a gate
- **THEN** the agent adds the check job to that workflow and its gate's dependencies, adds the three workflows and the labels, copies the script, and names the two maintainer actions

#### Scenario: Privileged job reads the head
- **WHEN** a produced workflow that runs with a writable token needs the head's documents
- **THEN** it checks out the base and reads them through the script's API snapshot, and no step fetches or checks out the head except the archive job's own checkout under the literal same-repository comparison

#### Scenario: Label plan checked before it is applied
- **WHEN** the label workflow applies the plan the script produced
- **THEN** it refuses and fails on any name outside the literal label taxonomy written in the workflow

#### Scenario: Fork pull request labeled
- **WHEN** the trigger label is applied to a pull request from a fork
- **THEN** the produced workflow mentions the author with the commands to run locally, removes the label, and pushes nothing

#### Scenario: Ready with an unarchived change
- **WHEN** a ready pull request still carries an unarchived related change
- **THEN** the produced check fails and names the change

#### Scenario: Fork pipeline in the parent project
- **WHEN** the user asks how to make the archive automation work for a fork's merge request on GitLab
- **THEN** the agent refuses the parent-project pipeline as a workaround and names the local commands instead

#### Scenario: GitLab install
- **WHEN** the user asks to install the automation in a GitLab project
- **THEN** the agent produces the jobs fragment and states that labels take effect on the next pipeline and that manual jobs replace comment commands

### Requirement: Handoff: the methodology skill
The agent SHALL route questions about whether or at which level to adopt spec-driven development, what a good specification is, or the generic loop to the methodology skill of the `sdd` catalog through the installing skill, and when the user declines SHALL answer from the loop facts this skill restates (package before tasks, archive before ready) without teaching the practice.

#### Scenario: Handoff offered
- **WHEN** the user asks whether the project should adopt spec-driven development at all
- **THEN** the agent offers the methodology skill through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the methodology skill
- **THEN** the agent answers the OpenSpec question at hand and says the practice-level question stays open

### Requirement: Handoff: the contract builder
The agent SHALL route setting or changing the project's rules — the approval mode, when a design is warranted, the archive executor, the request shape — to the spec workflow builder of the harness catalog through the installing skill, and when the user declines SHALL apply the contract as it stands or the defaults (by-hand executor, discussion-closed mode on the complete package) and say so.

#### Scenario: Handoff offered
- **WHEN** the user wants the label bot to become the project's recorded executor
- **THEN** the agent offers the spec workflow builder through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the builder
- **THEN** the agent installs the automation, applies the defaults for what the contract does not say, and names the contract update as remaining work

### Requirement: Script: spec_changes.py
The bundled script SHALL resolve a request's related changes from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer a `snapshot` subcommand that builds that document from the platform's REST API, listing the request's touched files and the related changes' documents without fetching or checking out the head, capping the files, bytes, and API calls it will read and refusing to report on a partial read of any kind — a cap reached, a truncated tree, or a document that is not decodable as text; a snapshot built for a different changes directory SHALL be refused rather than silently matched against none; `archive` SHALL accept only the git source, because it edits the working tree; SHALL offer `related`, `status`, `show`, `check`, `archive`, `labels`, and `snapshot` subcommands; `check` SHALL run the strict validator and fail on an unarchived related change (a warning with `--draft`) or, with `--all`, on any change outside the archive directory; `archive` SHALL refuse to run while any path the tool writes is uncommitted, SHALL archive every related change whose tasks are all ticked through the tool's archive command (with the spec-less flag for a marked change), refuse all of them when any has an open task — an unchecked box with no text is an open task, validate strictly afterwards, and change nothing on a repeated run; `show` SHALL print a change's documents inside a fence, truncate at `--max-chars` with a link, and distinguish a document that was never written from one the source does not hold; `labels` SHALL compute the desired archive-axis and progress-axis labels and the additions and removals against the current set, never touching the trigger label; every subcommand SHALL support `--help`, exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments.

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
