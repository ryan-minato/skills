## ADDED Requirements

### Requirement: Behavior: Approval scope is recorded as the approval package
The deposited contract SHALL state that the approval gate is exercised once the approval package is complete on the draft — the specification plus the design when one is warranted, with the rule the project settles for when a design is warranted — that the review covers the outcome description and the design's bounds (approach, technical constraints, preferences, rejected alternatives), never the task list or a step breakdown, that the task list follows the closing, and that the design is committed and therefore carries no secret or private data.

#### Scenario: Reading the approval gate
- **WHEN** a clean-context agent reads only the deposited contract
- **THEN** it can state which files make up the package, when the gate is exercised, that tasks are outside its scope, and what the design may and may not contain

### Requirement: Handoff: the framework skill for the selected approach
The builder SHALL route tool installation or adoption (step 3) and the installation of the archive automation and the validator's place in the check command (the second phase) to the framework skill for the selected approach in the `sdd` catalog through the installing skill, naming the skill by the approach it claims; when the user declines or no framework skill exists for the approach, the builder SHALL apply only the generic adoption rule (record the tree, run the tool's initializer with approval, diff and dispose of every touched file), SHALL record the by-hand executor and the checklist as the archive gate, and SHALL list the automation as remaining work in the hand-off.

#### Scenario: Handoff offered
- **WHEN** the selected approach is the spec-anchored change workflow and the builder reaches step 3
- **THEN** it offers the framework skill for that approach through the installing skill and prints no install command

#### Scenario: User declines
- **WHEN** the user declines the framework skill
- **THEN** the builder adopts the tool under the generic rule, records the by-hand executor, and lists the automation as remaining work

### Requirement: Behavior: Shape, archive executor, and author are settled with a reasoned recommendation
The builder SHALL ask the change request shape, the archive executor, the default specification author, the approval mode with the approval package and the rule for when a design is warranted, and the specification scope in the same round as the level and approach, each with one reasoned recommendation; SHALL derive the shape recommendation from the change propagation mode recorded in the project's workflow file when one exists; SHALL settle the approval mode for the two gates separately and recommend the discussion-closed mode for both by default; SHALL describe a recorded approval as an object the platform keeps — a review approval where one exists, a fixed-wording comment where none does — and SHALL derive the recommendation for it from whether someone outside the conversation must be able to verify by themselves that a named person accepted a named version, never from the team's size or its pipeline's maturity; SHALL state that a review approval cannot carry the specification gate wherever stale approvals are dismissed, so a project that wants a recorded approval usually wants it on the freeze alone; and SHALL check that the mode it recommends is available in the target, a repository whose author is its only reviewer having no review object to record; SHALL always record that archiving happens inside the change request once the deliberation on the finished implementation closes, and that it is the freeze the approval applies to; SHALL record the implementer as the executor and a maintainer on the contributor's branch for a request from a fork, and SHALL never offer an automation as the executor, because no platform token can push to a fork and a job that archives needs write access to the repository's contents for a benefit one command already gives; and SHALL recommend product-only domains with a spec-less change kind for the project's own harness, tooling, checks, workflows, and documents.

#### Scenario: Propagation recorded as Dependency
- **WHEN** the project's workflow file records dependency-style change propagation and the builder reaches its questioning round
- **THEN** the builder recommends the split shape and cites that line as the selecting fact

#### Scenario: CI exists
- **WHEN** the inspection finds a workflow or pipeline directory
- **THEN** the builder still records the implementer as the archive executor, and offers the framework skill's automation only for the check, the comment commands, and the status labels

#### Scenario: Automation asked for as the executor
- **WHEN** the user asks for a label or job that archives the request automatically
- **THEN** the builder declines, gives the two reasons, and records the implementer with the fork rule

#### Scenario: No CI
- **WHEN** the inspection finds no automation
- **THEN** the builder records the implementer as the executor and the request checklist as the gate

#### Scenario: Approval mode left unspecified
- **WHEN** neither the user nor any project file names an approval mechanism
- **THEN** the builder asks once whether anyone outside the conversation must verify the acceptance, recommends the discussion-closed mode on the complete package for both gates when nobody must, and records the answer

#### Scenario: An outside reader must verify the acceptance
- **WHEN** the project answers that an auditor or a contract party must be able to confirm who accepted which version
- **THEN** the builder recommends a recorded approval on the freeze gate only, keeps the specification gate conversational because implementation pushes dismiss a review approval, and records both

#### Scenario: Solo repository asks for a recorded approval
- **WHEN** the project's only reviewer is the author of its change requests
- **THEN** the builder says a review approval is unavailable there, offers the fixed-wording comment as the recorded form, and recommends the conversational mode unless an outside reader needs the record

#### Scenario: Tooling beside the product
- **WHEN** the inspection finds the project's own scripts, CI, and harness beside product code
- **THEN** the builder recommends product-only domains and a spec-less change kind for the rest, naming the tool's marker

## MODIFIED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project's spec-driven development rules must be initialized or improved, when a harness, its templates, or its tracker must be aligned with spec-driven development or a spec tool's layout, and when a delivered platform base awaits its specification shaping, and SHALL not cause it to load for writing a specification, choosing whether to adopt the practice, or installing one framework's automation.

#### Scenario: Harness alignment request
- **WHEN** the user says "our issue template and our openspec specs keep contradicting each other; make the harness match the tool"
- **THEN** the skill loads

#### Scenario: Delivered base awaits shaping
- **WHEN** the user says "our platform harness is built; now make the templates and the project skill follow our OpenSpec contract"
- **THEN** the skill loads

#### Scenario: Writing a specification
- **WHEN** the user says "write the spec for the export feature before we code it"
- **THEN** the skill does not load

#### Scenario: Framework automation request
- **WHEN** the user says "add the /spec comment commands and the status labels to our repository"
- **THEN** the skill does not load

### Requirement: Behavior: The deposited contract carries the new facts in platform vocabulary
The deposited specification contract SHALL state the change request shape, in-request archiving after the deliberation with its executor (the implementer, and a maintainer on the contributor's branch for a request from a fork), the freeze it creates and that approval applies to the frozen version, the automation the framework skill may install alongside it (a required check, comment commands, and status labels — nothing that archives or pushes), the default specification author, the approval mode for each of the two gates with its record (the closing of the discussion plus the reconciliation the agent performs, the archive commit standing as the record of the second closing; or, where the project names a recorded approval, the object that carries it and what it covers) exercised on the complete approval package, the specification scope with the spec-less change kind, the specification lines the change request body carries and the sections it reserves until ready, what the integration branch may hold, the framework skill the project uses, and the artifact operations — the categories of operation performed through the tool's commands (verified from its help, none quoted), the rule that hand edits are limited to the record's text, the validator with its strict mode, its place in the project's local check command, and when it runs (after each artifact edit, before publishing the draft, before ready, after archiving), or the structural check the project adopts when the tool has no validator — and SHALL use the project's platform vocabulary for work items, change requests, and automation, with no builder-only model noun appearing without its definition and no framework command or script named.

#### Scenario: Generated task list in the draft
- **WHEN** the deposited project skill states what the first draft carries
- **THEN** it says the approval package and that a task list the tool generated alongside it is pushed but marked as after-approval and kept out of the review, never that the draft carries no task list

#### Scenario: Entrypoint pointer
- **WHEN** the builder writes the entrypoint line that points at the contract
- **THEN** that line uses the platform's own words for the work item, carrying none of the builder's design vocabulary

#### Scenario: GitHub project deposit
- **WHEN** the builder deposits the contract for a project hosted on GitHub
- **THEN** the file names issues, pull requests, draft pull requests, and Actions workflows, states shape, executor, author, approval mode and package, and scope, names the framework skill, and contains no undefined model noun such as "tracked work" or "change request" and no tool command

#### Scenario: Reading the specification scope
- **WHEN** a clean-context agent reads only the deposited contract and is asked whether a change to the CI configuration needs a delta spec
- **THEN** it answers no and names the spec-less change kind and its record

#### Scenario: Tool without a validator
- **WHEN** the approach is a spec-first kit, an IDE's native spec files, or committed documents and the builder deposits the contract
- **THEN** the artifact operations section names the structural check the project adopts for its spec files or states that none exists, and quotes no command

### Requirement: Behavior: The platform base is shaped for the contract in a second phase
After the platform builder has delivered its paradigm-neutral base, the builder SHALL fill the base's extension slots for the evidenced platform from its own references and assets — the request template's specification block and checklist items worded for the package and the executor, the intake template's specification field, the project skill's take-work precondition, draft content (the complete approval package), reconciliation, and finish steps, the knowledge section, the sync rows, and the maintainer actions — locating each slot by heading, step, or field id, inserting without rewording base text, skipping a slot whose text is already present, and re-running the base's delivery checks; SHALL then hand the archive automation and the validator's place in the check command to the framework skill; and when no base exists the builder SHALL end its run after depositing the contract and say what remains.

#### Scenario: Base delivered on GitHub
- **WHEN** the contract is deposited, the GitHub base exists, and the builder runs
- **THEN** it inserts the specification lines into the pull request template, the task and feature forms, and the project skill, appends the knowledge section and sync rows, leaves no placeholder behind, and hands the automation to the framework skill

#### Scenario: Slot already filled
- **WHEN** the builder runs a second time on the same base
- **THEN** it finds every insertion already present and changes nothing

#### Scenario: Validator joins the check command
- **WHEN** the contract records a tool that ships a validator and the base delivered a local check command with a checks workflow that runs it
- **THEN** the builder hands the validator's strict run to the framework skill, which adds it to that check command, and edits no workflow file itself

#### Scenario: No base yet
- **WHEN** the project has a contract but no platform base
- **THEN** the builder ends after the contract deposit and names the platform builder and its own second phase as the remaining steps

### Requirement: Behavior: Take-work and the draft follow the change request shape on the platform
The project skill steps the builder inserts SHALL, under the combined shape, take a work item that has no specification by writing the change record's approval package, committing it, publishing it as a draft request, stopping, and waiting for the closed discussion or the approval comment per the contract's mode, and SHALL, under the split shape, escalate a work item whose specification request is not merged; a specification-only request SHALL reference the work item without closing it.

#### Scenario: Combined shape, no specification yet
- **WHEN** the contract records the combined shape and an agent takes a work item with an empty specification field
- **THEN** the inserted steps direct it to publish the change record with its complete approval package on a draft request and stop rather than escalating

#### Scenario: Split shape
- **WHEN** the contract records the split shape and the agent opens the specification request
- **THEN** the inserted steps make its body reference the work item with a non-closing reference

### Requirement: Behavior: Templates carry the specification block and checklist items
The request template lines the builder inserts SHALL add a specification block (the record link, the phase, one link per file of the approval package with the task list marked as after-approval, and the approval state with the exact closing or comment the contract fixes) and two checklist items — one confirming the package's discussion was closed or its approval commented before the task list (or that the request carries the specification only) and one confirming the record was archived or the specification updated inside the request before ready, by the executor the contract names — without changing the base's security item.

#### Scenario: Checklist check still passes
- **WHEN** a request body is built from the shaped template and every item is ticked
- **THEN** the project's checklist check passes

#### Scenario: Draft carries no implementation
- **WHEN** a body is built from the shaped template for a draft in the specification phase
- **THEN** the changes and validation sections hold the base's reserved line, the design is listed in the package, the task list is marked as after-approval, and the approval line names the closing or comment to wait for

### Requirement: Behavior: Closing the discussion reconciles the request before implementation
The project skill steps the builder inserts SHALL, under the discussion-closed mode, make the agent read the request's comments and review threads with their resolution state when the gate owner closes the discussion, list unresolved threads and requested adjustments missing from the package, ask the gate owner to confirm them, and start the task list and implementation only when nothing is open or the open items are confirmed.

#### Scenario: All threads resolved
- **WHEN** the gate owner closes the discussion and every thread is resolved
- **THEN** the inserted steps let the agent record the closing state and proceed to the task list

#### Scenario: Unresolved thread
- **WHEN** the gate owner closes the discussion while a review thread is unresolved
- **THEN** the inserted steps make the agent list the thread and confirm with the gate owner before implementing

#### Scenario: Requested adjustment missing from the record
- **WHEN** a comment asked for a record change that the record does not carry at closing time
- **THEN** the inserted steps make the agent name the gap and update the record before proceeding

## REMOVED Requirements

### Requirement: Behavior: Approval scope is recorded as outcome review
**Reason**: the gate now covers the complete approval package, design included.
**Migration**: "Approval scope is recorded as the approval package".

### Requirement: Behavior: Tool references distinguish fixed and project-defined archive operations
**Reason**: the builder no longer carries tool references; each framework skill in the `sdd` catalog owns its tool's artifacts, validator, and archive semantics.
**Migration**: the `sdd/openspec-workflow` and `sdd/spec-kit-workflow` domains.

### Requirement: Behavior: OpenSpec projects get a runnable archive job; other tools get design guidance
**Reason**: the after-merge archive job is withdrawn; in-request automation is installed by the framework skill.
**Migration**: the handoff to the framework skill and the `sdd/openspec-workflow` domain's automation requirement.

### Requirement: Script: archive_completed_changes.py
**Reason**: the job it served is withdrawn.
**Migration**: `Script: spec_changes.py` in the `sdd/openspec-workflow` domain.

### Requirement: Behavior: Shape, archive mode, and author are settled with a reasoned recommendation
**Reason**: an archive mode is no longer chosen; the question is who runs the archive inside the change request, so the requirement is renamed and rewritten.
**Migration**: "Shape, archive executor, and author are settled with a reasoned recommendation".
