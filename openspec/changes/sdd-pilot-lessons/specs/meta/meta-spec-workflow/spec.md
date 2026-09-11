## MODIFIED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project's spec-driven development rules must be initialized or improved, when a harness, its templates, or its tracker must be aligned with spec-driven development or a spec tool's layout, and when a delivered platform base awaits its specification shaping, and SHALL not cause it to load for writing a specification or choosing whether to adopt the practice.

#### Scenario: Harness alignment request
- **WHEN** the user says "our issue template and our openspec specs keep contradicting each other; make the harness match the tool"
- **THEN** the skill loads

#### Scenario: Delivered base awaits shaping
- **WHEN** the user says "our platform harness is built; now make the templates and the project skill follow our OpenSpec contract"
- **THEN** the skill loads

#### Scenario: Writing a specification
- **WHEN** the user says "write the spec for the export feature before we code it"
- **THEN** the skill does not load

### Requirement: Behavior: Shape, archive mode, and author are settled with a reasoned recommendation
The builder SHALL ask the change request shape, the archive mode, the default specification author, the approval mode, and the specification scope in the same round as the level and approach, each with one reasoned recommendation; SHALL derive the shape recommendation from the change propagation mode recorded in the project's workflow file when one exists; SHALL recommend the discussion-closed approval mode unless the user or the project asks for a blocking comment; and SHALL recommend product-only domains with a spec-less change kind for the project's own harness, tooling, checks, workflows, and documents.

#### Scenario: Propagation recorded as Dependency
- **WHEN** the project's workflow file records dependency-style change propagation and the builder reaches its questioning round
- **THEN** the builder recommends the split shape and cites that line as the selecting fact

#### Scenario: No automation can push
- **WHEN** the inspection finds no automation able to push to the integration branch
- **THEN** the builder recommends in-request archiving and records the missing capability as the reason

#### Scenario: Approval mode left unspecified
- **WHEN** neither the user nor any project file names an approval mechanism
- **THEN** the builder asks once whether a blocking comment is wanted, recommends the discussion-closed mode, and records the answer

#### Scenario: Tooling beside the product
- **WHEN** the inspection finds the project's own scripts, CI, and harness beside product code
- **THEN** the builder recommends product-only domains and a spec-less change kind for the rest, naming the tool's marker

### Requirement: Behavior: The deposited contract carries the new facts in platform vocabulary
The deposited specification contract SHALL state the change request shape, the archive mode with its serialization and push-rejection rules and the freeze that in-request archiving implies, the default specification author, the approval mode with its record (the closing of the discussion plus the reconciliation the agent performs, or the blocking comment's fixed text and what it covers), the specification scope with the spec-less change kind, the specification lines the change request body carries and the sections it reserves until ready, and what the integration branch may hold, and SHALL use the project's platform vocabulary for work items, change requests, and automation, with no builder-only model noun appearing without its definition.

#### Scenario: GitHub project deposit
- **WHEN** the builder deposits the contract for a project hosted on GitHub
- **THEN** the file names issues, pull requests, draft pull requests, and Actions workflows, states shape, archive mode, author, approval mode, and scope, and contains no undefined model noun such as "tracked work" or "change request"

#### Scenario: Reading the specification scope
- **WHEN** a clean-context agent reads only the deposited contract and is asked whether a change to the CI configuration needs a delta spec
- **THEN** it answers no and names the spec-less change kind and its record

### Requirement: Behavior: Tool references distinguish fixed and project-defined archive operations
The OpenSpec reference SHALL record both archive timings the tool supports, the tool's marker for a spec-less change and that the archive operation and the bundled script honor it (verified from the CLI's help, never quoted as a command), and that the split shape leaves approved change records on the integration branch; the Spec-Kit, Kiro, and committed-documents references SHALL state that automated archiving needs a project-defined completion criterion and post-processing step.

#### Scenario: Spec-Kit with automated archiving
- **WHEN** the selected approach is Spec-Kit and the user wants automated archiving
- **THEN** the builder asks the user to define the completion criterion and post-processing step before designing any job

#### Scenario: OpenSpec harness change
- **WHEN** the approach is OpenSpec and the user asks how a change to the CI configuration is recorded
- **THEN** the builder names the spec-less marker and says no domain is created

## ADDED Requirements

### Requirement: Behavior: The platform base is shaped for the contract in a second phase
After the platform builder has delivered its paradigm-neutral base, the builder SHALL fill the base's extension slots for the evidenced platform from its own references and assets — the request template's specification block and checklist items, the intake template's specification field, the project skill's take-work precondition, draft content, reconciliation, and finish steps, the archive workflow or job when the contract records automated archiving with OpenSpec, the knowledge section, the sync rows, and the maintainer action for the push path — locating each slot by heading, step, or field id, inserting without rewording base text, skipping a slot whose text is already present, and re-running the base's delivery checks; when no base exists the builder SHALL end its run after depositing the contract and say what remains.

#### Scenario: Base delivered on GitHub
- **WHEN** the contract is deposited, the GitHub base exists, and the builder runs
- **THEN** it inserts the specification lines into the pull request template, the task and feature forms, and the project skill, adds the archive workflow when the contract says automated with OpenSpec, appends the knowledge section and sync rows, and leaves no placeholder behind

#### Scenario: Slot already filled
- **WHEN** the builder runs a second time on the same base
- **THEN** it finds every insertion already present and changes nothing

#### Scenario: No base yet
- **WHEN** the project has a contract but no platform base
- **THEN** the builder ends after the contract deposit and names the platform builder and its own second phase as the remaining steps

### Requirement: Behavior: Take-work and the draft follow the change request shape on the platform
The project skill steps the builder inserts SHALL, under the combined shape, take a work item that has no specification by committing the change record to a draft request first, stopping, and waiting for the closed discussion or the approval comment per the contract's mode, and SHALL, under the split shape, escalate a work item whose specification request is not merged; a specification-only request SHALL reference the work item without closing it.

#### Scenario: Combined shape, no specification yet
- **WHEN** the contract records the combined shape and an agent takes a work item with an empty specification field
- **THEN** the inserted steps direct it to publish the change record on a draft request and stop rather than escalating

#### Scenario: Split shape
- **WHEN** the contract records the split shape and the agent opens the specification request
- **THEN** the inserted steps make its body reference the work item with a non-closing reference

### Requirement: Behavior: Templates carry the specification block and checklist items
The request template lines the builder inserts SHALL add a specification block (the record link, the phase, one link per record, and the approval state with the exact closing or comment the contract fixes) and two checklist items — one confirming the record's discussion was closed or its approval commented before implementation (or that the request carries the specification only) and one confirming the specification was updated or the record archived or completed for automated archiving — without changing the base's security item.

#### Scenario: Checklist check still passes
- **WHEN** a request body is built from the shaped template and every item is ticked
- **THEN** the project's checklist check passes

#### Scenario: Draft carries no implementation
- **WHEN** a body is built from the shaped template for a draft in the specification phase
- **THEN** the changes and validation sections hold the base's reserved line and the approval line names the closing or comment to wait for

### Requirement: Behavior: Closing the discussion reconciles the request before implementation
The project skill steps the builder inserts SHALL, under the discussion-closed mode, make the agent read the request's comments and review threads with their resolution state when the gate owner closes the discussion, list unresolved threads and requested adjustments missing from the record, ask the gate owner to confirm them, and start design and implementation only when nothing is open or the open items are confirmed.

#### Scenario: All threads resolved
- **WHEN** the gate owner closes the discussion and every thread is resolved
- **THEN** the inserted steps let the agent record the closing state and proceed to design

#### Scenario: Unresolved thread
- **WHEN** the gate owner closes the discussion while a review thread is unresolved
- **THEN** the inserted steps make the agent list the thread and confirm with the gate owner before implementing

#### Scenario: Requested adjustment missing from the record
- **WHEN** a comment asked for a record change that the record does not carry at closing time
- **THEN** the inserted steps make the agent name the gap and update the record before proceeding

### Requirement: Behavior: OpenSpec projects get a runnable archive job; other tools get design guidance
Under automated archiving with OpenSpec, the builder SHALL produce the platform's archive workflow or job from its asset — serialized, calling the project's copy of this builder's archive script, failing without retry on a rejected push — and SHALL record the push path as a maintainer action by owner type: on GitHub, the Actions app as a ruleset bypass actor for an organization-owned repository, and a deploy key with write access or a GitHub App installation token for a user-owned repository (the API refuses the Actions app there), noting that those pushes trigger workflows including the archive workflow itself; on GitLab, a project access token stored as a masked, protected variable plus an allowed-to-push entry on the protected default branch; for Spec-Kit, Kiro, and committed documents it SHALL design the job with the user from the same skeleton and copy no OpenSpec command.

#### Scenario: OpenSpec with automation
- **WHEN** the contract records OpenSpec and automated archiving
- **THEN** the produced workflow or job is serialized, calls the project's archive script, and its documentation records the maintainer action

#### Scenario: User-owned repository
- **WHEN** the repository is owned by a personal GitHub account
- **THEN** the guidance names a deploy key or a GitHub App as the push path, not the Actions app, and says those pushes re-trigger the workflow

#### Scenario: Protected branch refuses the token
- **WHEN** a GitLab project access token exists but the protected branch has no allowed-to-push entry for it
- **THEN** the guidance says the push is refused and names the entry as the maintainer action

#### Scenario: Spec-Kit with automation
- **WHEN** the contract records Spec-Kit and automated archiving
- **THEN** the builder asks for the completion criterion and post-processing step and produces no OpenSpec command

### Requirement: Script: archive_completed_changes.py
The bundled script SHALL archive every OpenSpec change whose task list has at least one completed task and no open task, merging its delta into the main specs and validating strictly; SHALL archive a change whose record marks it spec-less (`skip_specs: true`) without touching the main specs; SHALL support `--help` and `--dry-run`; SHALL exit 0 with no output when nothing is completed; SHALL change nothing on an identical repeated run; and SHALL exit 2 with a diagnostic on bad arguments.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming `--dry-run` and the spec-less case and exits 0

#### Scenario: Representative run
- **WHEN** a change with every task ticked exists and the script runs
- **THEN** the change moves under the archive directory, the main specs carry its delta, and strict validation passes

#### Scenario: Spec-less change
- **WHEN** a change whose record sets `skip_specs: true` has every task ticked and no delta spec
- **THEN** it moves under the archive directory, the main specs are unchanged, and strict validation passes

#### Scenario: Nothing completed
- **WHEN** the script runs with `--dry-run` in a repository whose in-flight changes all have open tasks
- **THEN** it exits 0 and prints nothing

#### Scenario: Repeated run
- **WHEN** the identical command runs a second time after the representative run
- **THEN** nothing changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option
