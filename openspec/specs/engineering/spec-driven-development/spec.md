# engineering/spec-driven-development Specification

## Purpose
Governs what an agent that loaded the `spec-driven-development` skill observably does when it settles how specifications and tracked work interact: the change request shape, the archive mode, the approval record, the timing of each step, the scope of specification review, and harness design without a builder.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load for questions about how issues, pull or merge requests, and specifications fit together under spec-driven development, and SHALL not cause it to load for a request to build platform intake forms or CI.

#### Scenario: Lifecycle question
- **WHEN** the user asks "how should issues and PRs work now that we use OpenSpec?"
- **THEN** the skill loads

#### Scenario: Harness build request
- **WHEN** the user asks "set up GitHub issue forms and CI for us"
- **THEN** the skill does not load

### Requirement: Behavior: Change request shape is recommended from the project's consumers
When asked how specifications relate to pull or merge requests, the agent SHALL recommend exactly one change request shape — combined (one change request carries the specification and its implementation) or split (the specification merges through its own change request first) — name the fact that selected it (whether consumers depend on a stable contract, or the recorded change propagation mode), and offer the other shape as the recorded deviation, leaving the decision to the user.

#### Scenario: Library with downstream consumers
- **WHEN** the user describes a versioned library whose consumers depend on its API and asks how specs and pull requests should relate
- **THEN** the agent recommends the split shape, cites the consumer contract as the selecting fact, and offers the combined shape as the deviation without deciding for the user

#### Scenario: Feature-driven application
- **WHEN** the user describes an application delivered feature by feature with no external consumers of a contract
- **THEN** the agent recommends the combined shape and names the absence of a consumer contract as the selecting fact

### Requirement: Behavior: Specification review happens on the published draft with a recorded approval
The agent SHALL place specification review on the draft change request (combined) or on the specification change request (split), SHALL have the approval recorded as a comment by the gate owner naming the approved commit, and SHALL warn that a platform review-approval state is dismissed by later pushes.

#### Scenario: Where the spec is reviewed
- **WHEN** a user on the combined shape asks where the team reviews the specification
- **THEN** the agent answers the draft change request, says approval is a comment from the gate owner naming the approved commit, and warns that platform review approvals do not survive the implementation pushes

### Requirement: Behavior: The draft is published as soon as the specification is written
The agent SHALL publish the change record as a draft change request (or as the specification change request) as soon as the specification is written and clarified, before planning or tasks, and SHALL not start planning or implementation before the approval is recorded.

#### Scenario: Planning before publication
- **WHEN** the specification is written and the user asks the agent to write the plan and tasks next
- **THEN** the agent first publishes (or asks to publish) the draft carrying the specification and states that plan and tasks follow the recorded approval

### Requirement: Behavior: Specification review examines the outcome, never the tasks
The agent SHALL describe specification review as covering outcome descriptions — goals and scope, terminology and domain model, behavior, invariants, constraints and rules, states and transitions, interface and data contracts, exceptions and edge cases, security and permissions, metrics and acceptance criteria, each as the project needs — and SHALL exclude tasks and design from the approval gate.

#### Scenario: Tasks offered for review
- **WHEN** the user asks the agent to include the task list in the specification review
- **THEN** the agent declines, explains that tasks describe how the outcome is built and belong to the implementer after approval, and lists what the review does examine

### Requirement: Behavior: Archive mode is recommended from the automation available
The agent SHALL recommend automated archiving (a serialized, idempotent job on the integration branch after merge) when the remote can run automation that may push to the integration branch, and in-request archiving (before the change request is marked ready) otherwise, and SHALL state the serialization, idempotence, and push-rejection rules for the automated mode.

#### Scenario: Automation cannot push
- **WHEN** the project has CI but its automation identity is not allowed to push to the protected default branch
- **THEN** the agent recommends in-request archiving and names the missing push permission as the reason

#### Scenario: Automation available
- **WHEN** the project's CI may push to the default branch
- **THEN** the agent recommends automated archiving and states that the job runs one at a time, rescans every completed change on each run, and fails without retry when its push is rejected

### Requirement: Behavior: Tracked work opens at requirement time without acceptance criteria
The agent SHALL open (or direct the user to open) the work item when the requirement appears, carrying the raw requirement, owner, and priority, and SHALL refuse to restate acceptance criteria in it; the item links the change record once that record exists.

#### Scenario: Acceptance pasted into the issue
- **WHEN** the user asks to paste the acceptance criteria into the work item
- **THEN** the agent refuses the restatement and provides the link form pointing at the change record's scenarios

### Requirement: Handoff: the harness builder for spec workflows
The agent SHALL route harness alignment to the harness builder for spec workflows through the installing skill; when the user declines or lacks it, the agent SHALL record the shape, archive mode, timing, approval owner, and navigation rule in the project's knowledge base, add the named specification lines to the project's existing intake and change-request templates, and list the harness build as remaining work, without creating forms, checks, CI, or a project skill.

#### Scenario: Handoff offered
- **WHEN** the level, tool, shape, and archive mode are settled and the harness must state them
- **THEN** the agent offers the harness builder for spec workflows through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user says not to install any builder
- **THEN** the agent produces the knowledge entry and template lines itself and lists the platform harness build as remaining work

### Requirement: Script: archive_completed_changes.py
The bundled script SHALL archive every OpenSpec change whose task list has at least one completed task and no open task, merging its delta into the main specs and validating strictly; SHALL support `--help` and `--dry-run`; SHALL exit 0 with no output when nothing is completed; SHALL change nothing on an identical repeated run; and SHALL exit 2 with a diagnostic on bad arguments.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming `--dry-run` and exits 0

#### Scenario: Representative run
- **WHEN** a change with every task ticked exists and the script runs
- **THEN** the change moves under the archive directory, the main specs carry its delta, and strict validation passes

#### Scenario: Nothing completed
- **WHEN** the script runs with `--dry-run` in a repository whose in-flight changes all have open tasks
- **THEN** it exits 0 and prints nothing

#### Scenario: Repeated run
- **WHEN** the identical command runs a second time after the representative run
- **THEN** nothing changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option
