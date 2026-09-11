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
When the project's specification contract records a change request shape, the agent SHALL apply it without re-asking; when no contract exists, the agent SHALL apply the combined shape by default and name the spec workflow builder of the harness catalog as the way to settle and record another; when the user asks directly how specifications should relate to pull or merge requests, the agent SHALL explain both shapes and the fact that selects each (whether consumers depend on a stable contract) and SHALL say the decision is recorded by that builder, not by this skill.

#### Scenario: Library with downstream consumers
- **WHEN** the user describes a versioned library whose consumers depend on its API and asks how specs and pull requests should relate, and no contract exists
- **THEN** the agent explains that the split shape fits a consumer contract, names the fact, offers the combined shape as the alternative, and says the harness builder records the choice

#### Scenario: Feature-driven application
- **WHEN** the user describes an application delivered feature by feature with no external consumers of a contract, and no contract exists
- **THEN** the agent applies the combined shape, names the absence of a consumer contract as the reason, and says the harness builder records it

#### Scenario: Contract already records the shape
- **WHEN** the project's contract records the split shape and the user asks the agent to open the change request
- **THEN** the agent opens a specification change request carrying only the record, without asking which shape to use

### Requirement: Behavior: Specification review happens on the published draft with a recorded approval
The agent SHALL place specification review on the draft change request (combined) or on the specification change request (split), SHALL apply the approval mode the contract records — discussion-closed: the gate owner discusses on the request and closes the discussion in conversation, and the approval record is that closing plus the request's discussion state; blocking: the gate owner's fixed-wording comment on the draft, naming no commit, covering the record as of the last push before it, with a later push to the record needing a fresh comment unless the gate owner decided that push in conversation — and SHALL warn that a platform review-approval state is not the record in either mode because later pushes dismiss it.

#### Scenario: Where the spec is reviewed
- **WHEN** a user on the combined shape asks where the team reviews the specification
- **THEN** the agent answers the draft change request, states the contract's approval mode (or the discussion-closed default), and warns that platform review approvals do not survive the implementation pushes

#### Scenario: Push to the record after a blocking approval
- **WHEN** the contract records the blocking mode, the approval comment exists, and the author then pushes a change to the record that the gate owner did not decide in conversation
- **THEN** the agent asks for a fresh approval comment before planning

#### Scenario: Narrowing decided by the gate owner
- **WHEN** the gate owner decides in conversation to drop a scenario from an approved record and the author pushes that narrowing
- **THEN** the agent proceeds without a fresh comment and records the decision in the request

### Requirement: Behavior: The draft is published as soon as the specification is written
The agent SHALL publish the change record as a draft change request (or as the specification change request) as soon as the specification is written and clarified, before planning or tasks, and SHALL then stop: it SHALL not start design, tasks, or implementation until the discussion is closed (discussion-closed mode) or the approval comment exists (blocking mode).

#### Scenario: Planning before publication
- **WHEN** the specification is written and the user asks the agent to write the plan and tasks next
- **THEN** the agent first publishes (or asks to publish) the draft carrying the specification and states that plan and tasks follow the closed discussion or the approval comment

#### Scenario: Waiting after publication
- **WHEN** the draft is published and nobody has closed the discussion or commented an approval
- **THEN** the agent does not write design or tasks and says what it is waiting for

### Requirement: Behavior: Specification review examines the outcome, never the tasks
The agent SHALL describe specification review as covering outcome descriptions — goals and scope, terminology and domain model, behavior, invariants, constraints and rules, states and transitions, interface and data contracts, exceptions and edge cases, security and permissions, metrics and acceptance criteria, each as the project needs — and SHALL exclude tasks and design from the approval gate.

#### Scenario: Tasks offered for review
- **WHEN** the user asks the agent to include the task list in the specification review
- **THEN** the agent declines, explains that tasks describe how the outcome is built and belong to the implementer after approval, and lists what the review does examine

### Requirement: Behavior: Archive mode is recommended from the automation available
The agent SHALL archive per the mode the contract records; when no contract exists it SHALL default to in-request archiving and say that automated archiving needs a push path the harness builder records; SHALL state that in-request archiving happens before review, so the archived record is frozen and review-driven corrections go to the request's validation section or a follow-up change; and SHALL prefer automated archiving wherever a push path exists.

#### Scenario: Automation cannot push
- **WHEN** the project has CI but its automation identity is not allowed to push to the protected default branch, and no contract records a mode
- **THEN** the agent archives inside the request and names the missing push path as the reason

#### Scenario: Automation available
- **WHEN** the contract records automated archiving because the project's CI may push to the default branch
- **THEN** the agent leaves the completed change for the archive job and states that the job runs one at a time, rescans every completed change on each run, and fails without retry when its push is rejected

#### Scenario: Review finds a defect after in-request archiving
- **WHEN** review finds an error in a record already archived inside the request
- **THEN** the agent records the correction in the request's validation section (or opens a follow-up change) and does not edit the archived record

### Requirement: Behavior: Tracked work opens at requirement time without acceptance criteria
The agent SHALL open (or direct the user to open) the work item when the requirement appears, carrying the raw requirement, owner, and priority, and SHALL refuse to restate acceptance criteria in it; the item links the change record once that record exists.

#### Scenario: Acceptance pasted into the issue
- **WHEN** the user asks to paste the acceptance criteria into the work item
- **THEN** the agent refuses the restatement and provides the link form pointing at the change record's scenarios

### Requirement: Handoff: the harness builder for spec workflows
The agent SHALL route setting up or improving a project's specification rules to the spec workflow builder of the harness catalog through the installing skill, which installs that catalog whole; when the user declines or lacks it, the agent SHALL record the defaults it applies (combined shape, discussion-closed approval, in-request archiving unless automation can push, product-only domains with spec-less repository changes, a request body that carries no implementation until ready) in the project's knowledge base and list the harness build as remaining work, without editing templates, forms, checks, automation, or a project skill.

#### Scenario: Handoff offered
- **WHEN** the user asks to set up or improve the project's spec-driven rules
- **THEN** the agent offers the spec workflow builder of the harness catalog through the installing skill, says the catalog installs whole, and prints no install command

#### Scenario: User declines
- **WHEN** the user says not to install any builder
- **THEN** the agent writes the defaults into the knowledge base, edits no template or harness file, and lists the harness build as remaining work

### Requirement: Behavior: Level, tool, and lifecycle facts are read from the contract, defaulted when absent
The agent SHALL take the level, tool, change request shape, approval mode, archive mode, and specification scope from the project's specification contract when one exists; when none exists it SHALL apply the documented defaults, say so, and name the spec workflow builder of the harness catalog as the way to settle and record them; it SHALL not run a questioning round of its own.

#### Scenario: Project with a contract
- **WHEN** the project's knowledge base carries a specification contract and the user starts a change
- **THEN** the agent applies the contract's facts and asks none of them again

#### Scenario: No contract
- **WHEN** the project has no specification contract and the user asks what happens next after the draft is published
- **THEN** the agent says it stops until the discussion is closed (the default), names the harness builder for changing that, and asks no questioning round

### Requirement: Behavior: Closing the discussion reconciles the request before implementation
When the gate owner says in conversation that the discussion on the draft is closed, the agent SHALL read the request's comments and review threads with their resolution state, SHALL check for unresolved threads, adjustments requested in the discussion that the record does not yet carry, and conclusions that contradict each other, SHALL list every open item and ask the gate owner to confirm before proceeding, and SHALL start design, tasks, and implementation only when nothing is open or the gate owner confirmed the open items; the agent SHALL record the closing state in the request's specification block.

#### Scenario: All threads resolved
- **WHEN** the gate owner closes the discussion and every thread is resolved and every requested adjustment is in the record
- **THEN** the agent records the closing state and starts design and tasks

#### Scenario: Unresolved thread
- **WHEN** the gate owner closes the discussion while a review thread is unresolved
- **THEN** the agent lists the thread, asks the gate owner to confirm or resolve it, and does not implement

#### Scenario: Requested adjustment missing from the record
- **WHEN** a discussion comment asked for a scenario to change and the record still carries the old scenario at closing time
- **THEN** the agent names the gap, updates the record (or asks whether the request was withdrawn), and only then continues

#### Scenario: Adjustment requested mid-discussion
- **WHEN** the gate owner directs a record change in conversation while the discussion is open
- **THEN** the agent updates the record, publishes it through the project's publish gate, and keeps waiting for the closing

### Requirement: Behavior: The change request body navigates to the record and carries no implementation until ready
The agent SHALL fill the change request body per the project's template and, absent one, per this default: an opening paragraph stating the goal (not the work), a section stating the value, a specification block with a link to the change record on the branch, the phase, one link per record, and the approval state in the contract's mode, related work with the closing reference, and changes and validation reserved until the request is marked ready — changes as permalinks to the commits (the exact lines for a local change, the whole file or directory for a broad one), validation naming each scenario with its result and linking the plan; the agent MAY add sections beyond these and SHALL pass every section through the project's publish gate.

#### Scenario: Draft body in the specification phase
- **WHEN** the user asks for the draft's description for a proposed change
- **THEN** the output opens with the goal, links the record, states the phase and the approval state, and leaves changes and validation reserved with no implementation detail

#### Scenario: Implementation offered for the draft
- **WHEN** the user asks to paste the task list or a diff summary into the draft before the discussion is closed
- **THEN** the agent declines and keeps those sections reserved until ready

### Requirement: Behavior: Specification domains cover the product; changes to the project's own harness are spec-less
The agent SHALL treat specification domains as describing the product a project delivers, SHALL treat a change to the project's own harness, tooling, checks, workflows, or documents as a spec-less change kind carried by the tool's marker (OpenSpec: `skip_specs: true` with a proposal, design, and tasks and no delta spec), SHALL have tracked work link such a change the same way, and SHALL refuse to create a domain for tooling.

#### Scenario: CI workflow change
- **WHEN** the user asks whether changing the lint workflow needs a delta spec
- **THEN** the agent answers no, names the spec-less change kind and its record, and says tracked work links it

#### Scenario: Domain requested for tooling
- **WHEN** the user asks to write a domain spec for the build scripts
- **THEN** the agent refuses and offers the spec-less change kind instead

### Requirement: Behavior: Baseline scenarios are verified and untouched defects are filed, never absorbed
When a change first creates a domain and the project's schema requires a baseline block, the agent SHALL verify its scenarios like any other; when a baseline scenario fails for behavior the change does not touch, the agent SHALL file it as a separate defect and narrow the record to what is verified, and SHALL never widen the change silently.

#### Scenario: Failing baseline scenario
- **WHEN** a baseline scenario fails for a phrasing the change does not touch
- **THEN** the agent files the failure separately with the evidence, narrows the record, and continues the change without absorbing the fix

### Requirement: Behavior: Spec artifacts are created through the tool's commands and validated programmatically
When the project uses a specification tool, the agent SHALL perform every operation the tool has a command for — initializing, creating a change or feature record, validating, archiving — through that command after verifying it from the tool's help and current documentation, SHALL not create the tool's directory tree or generated files by hand, SHALL limit hand edits to the requirement text itself, SHALL run the tool's validator (strict mode where it exists) after each artifact edit, before publishing the draft, before marking ready, and after archiving, and SHALL fix what it reports before proceeding; when the tool has no validator the agent SHALL check the artifacts against the tool's documented structure and say that no programmatic check exists; a passing validation proves structure, not content, and never replaces the specification review.

#### Scenario: Tool scaffolds the change
- **WHEN** the project uses OpenSpec and the user asks for the change record of a new feature
- **THEN** the agent reads the tool's help, creates the change with the tool's command, and creates no directory or metadata file by hand

#### Scenario: Validator available
- **WHEN** the change record is written and the draft is about to be published
- **THEN** the agent runs the tool's validator in strict mode, fixes what it reports, and publishes only when it passes

#### Scenario: Tool without a validator
- **WHEN** the project keeps committed specification documents and the record is written
- **THEN** the agent checks the documents against the project's recorded structure, says that no programmatic validator exists, and names the structural check the contract records if any

#### Scenario: Hand-written record offered
- **WHEN** the user asks the agent to write the change directory and its files directly because it is faster
- **THEN** the agent declines, names the metadata the tool's command produces, and creates the record through the command
