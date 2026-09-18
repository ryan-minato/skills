# sdd/spec-kit-workflow Specification

## Purpose
Governs what an agent that loaded the `spec-kit-workflow` skill observably does when a project runs Spec-Kit features through pull or merge requests: the approval package, the completion criterion before ready, the comment commands and progress labels, and the automation it installs per platform.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project that runs Spec-Kit asks how a feature's specification and plan pass approval, what must be complete before the request is ready, or how to use or install the `/spec` comment commands, the progress labels, and the feature check in CI, and SHALL not cause it to load for running the kit's own commands on one feature, for choosing a spec tool, or for a project that runs another framework.

#### Scenario: Approval question
- **WHEN** the user says "does the plan.md of our Spec-Kit feature need to be approved before we write tasks?"
- **THEN** the skill loads

#### Scenario: Automation install
- **WHEN** the user says "add progress labels and a feature check for our Spec-Kit repository"
- **THEN** the skill loads

#### Scenario: Kit command request
- **WHEN** the user says "run the specify command to start the export feature"
- **THEN** the skill does not load

#### Scenario: Other framework
- **WHEN** the user says "our OpenSpec change is done; how does it get archived?"
- **THEN** the skill does not load

### Requirement: Behavior: The approval package is the specification and the plan
The agent SHALL treat the feature's `spec.md` and `plan.md` together as the approval package the draft opens with (the plan bounds the approach; it is not a step list), SHALL treat `tasks.md` as after-approval material even when the kit generated it, SHALL check the feature directory against the kit's template headings and say that no programmatic validator exists, and SHALL say that the kit's feature script creates a numbered directory and no branch.

#### Scenario: Reviewer's set
- **WHEN** the user asks what the reviewer is handed at the gate
- **THEN** the agent answers the specification and the plan, and says the task list follows the closing

#### Scenario: Plan written as steps
- **WHEN** the plan reads as an ordered implementation procedure
- **THEN** the agent rewrites it as approach, constraints, and preferences before publishing the draft

### Requirement: Behavior: Completion replaces archiving, and the contract says what locks the specification
The agent SHALL state that Spec-Kit has no archive operation, SHALL require every task of every feature the request touches to be ticked before the request is marked ready for the deliberation on the finished implementation, and SHALL refuse to tick a task whose verification did not run. Because the kit has no archive, it has no lock event: the agent SHALL say that what marks the specification frozen before approval is whatever the project's contract declares — by default the deliberation closing with no further change to the specification or the plan — and that nothing in the kit enforces it. The agent SHALL name the project's rule for updating the living specification when the level is spec-anchored.

#### Scenario: Ready with an open task
- **WHEN** the user asks to mark the request ready while a touched feature's task list has an open item
- **THEN** the agent refuses, names the task, and says the check would fail

#### Scenario: Archive asked
- **WHEN** the user asks how to archive the feature
- **THEN** the agent says the kit archives nothing, completion is the ticked task list, and a spec-anchored project updates its living specification per its rule

#### Scenario: What locks the specification
- **WHEN** the user asks what the approval applies to, given that nothing is archived
- **THEN** the agent names the contract's lock declaration, says the default is the deliberation closing with the specification and the plan unchanged, and says no mechanism enforces it

### Requirement: Behavior: The automation is installed per platform from the skill's assets
When asked to install the automation on GitHub, the agent SHALL produce from its assets a check job that joins the base's existing checks workflow and its gate (required files present for every touched feature; an open task as a warning on a draft and a failure when ready), SHALL configure that job with the change request shape the contract records so that under the combined shape no request merges unfinished while under the split shape the one request that implements nothing — the specification request — is admitted and any request that completed a task is not, the comment-command workflow (`/spec show` and `/spec status` over touched features), and the progress-label workflow, SHALL add the three progress labels to the project's label file, SHALL copy `scripts/spec_kit_features.py` into the project's scripts, SHALL grant each job only the scopes its own reads and writes need, remembering that a permission left out of the block is none, SHALL keep every workflow fork-safe by one structural rule — no object authored by the request reaches a privileged runner: scripts run from the base ref, no job checks the head out or fetches it, and the head is read through the script's API snapshot, SHALL restrict the comment command to the collaborator associations so nobody without write access can start a privileged run, SHALL explain that a fork's unprivileged run holds a read-only token and no secrets and that the setting which would grant write exists for private repositories only, which is why labelling and replying need a privileged trigger at all, and SHALL say that no job archives or pushes, both because the kit has no archive operation and because the executor of a freeze is a person; on GitLab the agent SHALL produce the jobs fragment, state its limitations, and refuse running a fork's pipeline in the parent project as a workaround, because that runs the fork's configuration with the parent's token.

#### Scenario: GitHub install
- **WHEN** the user asks to install the automation in a GitHub repository whose base has a checks workflow with a gate
- **THEN** the agent adds the check job to that workflow and its gate's dependencies, adds the two privileged workflows and the labels, copies the script, and says nothing it installs archives or pushes

#### Scenario: Combined shape, nothing implemented
- **WHEN** a ready request on a combined-shape project touches a feature whose task list has no completed task
- **THEN** the produced check fails it, because a request that implemented nothing is unfinished rather than exempt

#### Scenario: Split shape, the specification request
- **WHEN** a ready request on a split-shape project carries the specification and the plan with no completed task
- **THEN** the produced check admits it, and fails a later request of the same project that completed one and left the rest open
#### Scenario: Privileged job reads the head
- **WHEN** a produced workflow that runs with a writable token needs a touched feature's documents
- **THEN** it checks out the base and reads them through the script's API snapshot, and no step of any produced workflow fetches or checks the head out

#### Scenario: Ready with an open task
- **WHEN** a ready pull request touches a feature whose task list has an open item
- **THEN** the produced check fails and names the feature and the task

#### Scenario: Command invoked by the request's author
- **WHEN** a comment command is posted by someone who is not a collaborator, including the request's own author
- **THEN** the produced workflow does not run

### Requirement: Handoff: the methodology skill
The agent SHALL route questions about whether or at which level to adopt spec-driven development, what a good specification is, or the generic loop to the methodology skill of the `sdd` catalog through the installing skill, and when the user declines SHALL answer from the loop facts this skill restates (package before tasks, completion before ready) without teaching the practice.

#### Scenario: Handoff offered
- **WHEN** the user asks whether the project should adopt spec-driven development at all
- **THEN** the agent offers the methodology skill through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the methodology skill
- **THEN** the agent answers the Spec-Kit question at hand and says the practice-level question stays open

### Requirement: Handoff: the contract builder
The agent SHALL route setting or changing the project's rules — the approval mode, the completion criterion, the request shape — to the spec workflow builder of the harness catalog through the installing skill, and when the user declines SHALL apply the contract as it stands or the defaults and say so.

#### Scenario: Handoff offered
- **WHEN** the user wants the ticked task list to become the project's recorded completion rule
- **THEN** the agent offers the spec workflow builder through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the builder
- **THEN** the agent installs the automation, applies the defaults, and names the contract update as remaining work

### Requirement: Script: spec_kit_features.py
The bundled script SHALL resolve a request's touched features (numbered directories under the kit's specs directory whose files the request touches) from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer a `snapshot` subcommand that builds that document from the platform's REST API without fetching or checking out the head, capping the files, bytes, and API calls it will read and refusing to report on a partial read of any kind — a cap reached, a truncated tree, or a document that is not decodable as text; a snapshot built for a different specs directory SHALL be refused rather than silently matched against none; SHALL offer `related`, `status`, `show`, `check`, `labels`, and `snapshot` subcommands; `check` SHALL fail when a touched feature lacks its specification or plan, or when a touched feature has an open task and the request is ready (a warning with `--draft`); an unchecked box with no text is an open task; `show` SHALL print a feature's documents inside a fence and truncate at `--max-chars` with a link; `labels` SHALL compute the desired progress label and the additions and removals against the current set; every subcommand SHALL support `--help`, exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming the six subcommands and exits 0

#### Scenario: Partial read refused
- **WHEN** the `snapshot` command reaches a cap, meets a truncated tree, or finds a document it cannot decode as text
- **THEN** it exits 1 naming the path and reports nothing, so no command runs on data it does not hold

#### Scenario: Snapshot source matches the git source
- **WHEN** `snapshot` runs against a pull request and `status`, `show`, and `labels` run against the resulting document
- **THEN** their output is identical to the same subcommands run with `--base` and `--head` over a clone of that pull request

#### Scenario: Representative run
- **WHEN** a touched feature has its specification, plan, and a fully ticked task list and `check --base <base> --head <head>` runs
- **THEN** it exits 0, and `status` reports the feature as done

#### Scenario: Repeated run
- **WHEN** the identical `check` command runs a second time
- **THEN** the output is identical and nothing in the tree changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option

#### Scenario: Missing plan
- **WHEN** a touched feature has a specification but no plan
- **THEN** `check` exits 1 and names the feature and the missing file
