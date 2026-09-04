## Purpose

Governs what an agent that loaded the `meta-spec-workflow` builder observably asks, records, and deposits about the change request shape, the archive mode, the specification author, the approval record, and the vocabulary of the deposited contract.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a harness, its templates, or its tracker must be aligned with spec-driven development or a spec tool's layout, and SHALL not cause it to load for writing a specification or choosing whether to adopt the practice.

#### Scenario: Harness alignment request
- **WHEN** the user says "our issue template and our openspec specs keep contradicting each other; make the harness match the tool"
- **THEN** the skill loads

#### Scenario: Writing a specification
- **WHEN** the user says "write the spec for the export feature before we code it"
- **THEN** the skill does not load

### Requirement: Behavior: Shape, archive mode, and author are settled with a reasoned recommendation
The builder SHALL ask the change request shape, the archive mode, and the default specification author in the same round as the level and approach, each with one reasoned recommendation, and SHALL derive the shape recommendation from the change propagation mode recorded in the project's workflow file when one exists.

#### Scenario: Propagation recorded as Dependency
- **WHEN** the project's workflow file records dependency-style change propagation and the builder reaches its questioning round
- **THEN** the builder recommends the split shape and cites that line as the selecting fact

#### Scenario: No automation can push
- **WHEN** the inspection finds no automation able to push to the integration branch
- **THEN** the builder recommends in-request archiving and records the missing capability as the reason

### Requirement: Behavior: The deposited contract carries the new facts in platform vocabulary
The deposited specification contract SHALL state the change request shape, the archive mode with its serialization and push-rejection rules, the default specification author, where the approval is recorded, and what the integration branch may hold, and SHALL use the project's platform vocabulary for work items, change requests, and automation, with no builder-only model noun appearing without its definition.

#### Scenario: GitHub project deposit
- **WHEN** the builder deposits the contract for a project hosted on GitHub
- **THEN** the file names issues, pull requests, draft pull requests, and Actions workflows, states shape, archive mode, author, and approval record, and contains no undefined model noun such as "tracked work" or "change request"

### Requirement: Behavior: Approval scope is recorded as outcome review
The deposited contract SHALL state that the approval gate is exercised as soon as the specification is published as a draft and reviews outcome descriptions, never design or tasks.

#### Scenario: Reading the approval gate
- **WHEN** a clean-context agent reads only the deposited contract
- **THEN** it can state when the gate is exercised and that tasks are outside its scope

### Requirement: Behavior: Tool references distinguish fixed and project-defined archive operations
The OpenSpec reference SHALL record both archive timings the tool supports and that the split shape leaves approved change records on the integration branch; the Spec-Kit, Kiro, and committed-documents references SHALL state that automated archiving needs a project-defined completion criterion and post-processing step.

#### Scenario: Spec-Kit with automated archiving
- **WHEN** the selected approach is Spec-Kit and the user wants automated archiving
- **THEN** the builder asks the user to define the completion criterion and post-processing step before designing any job
