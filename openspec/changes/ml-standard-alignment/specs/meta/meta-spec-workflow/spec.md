## MODIFIED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project's spec-driven development rules must be initialized or improved, when a harness, its templates, or its tracker must be aligned with spec-driven development or a spec tool's layout, when a delivered platform base awaits its specification shaping, and when a research repository needs a research-task contract whose specification is an objective and an evaluation rather than requirements, and SHALL not cause it to load for writing a specification or choosing whether to adopt the practice.

#### Scenario: Harness alignment request
- **WHEN** the user says "our issue template and our openspec specs keep contradicting each other; make the harness match the tool"
- **THEN** the skill loads

#### Scenario: Delivered base awaits shaping
- **WHEN** the user says "our platform harness is built; now make the templates and the project skill follow our OpenSpec contract"
- **THEN** the skill loads

#### Scenario: Research repository contract
- **WHEN** the user says "this repo runs ML experiments, not features — set up how a research task is specified, approved, and closed"
- **THEN** the skill loads

#### Scenario: Writing a specification
- **WHEN** the user says "write the spec for the export feature before we code it"
- **THEN** the skill does not load

## ADDED Requirements

### Requirement: Behavior: A research repository is offered the research-task approach
When the workflow contract records the research profile, the builder SHALL offer, as its recommended approach, the research-task protocol on the spec-anchored change workflow tool with a project-local `research-task` schema — a spec that carries an Objective and an Evaluation (Context, Search Scope, Constraints, Completion Condition, Hypotheses as needed) rather than requirements and scenarios — SHALL state that research tasks hold no domain while any product code keeps domains under the default schema, and SHALL keep a spec tool the project already runs.

#### Scenario: Research profile recorded
- **WHEN** the workflow file records the research profile and no spec tool exists
- **THEN** the approach question recommends the research-task protocol on the change workflow tool with the research-task schema and names the fact that selected it

#### Scenario: Software profile recorded
- **WHEN** the workflow file records a continuous-product profile
- **THEN** the approach question does not mention the research-task schema

### Requirement: Behavior: The research contract is deposited with its own gate, evolution, and closing rules
When the research-task approach is settled, the deposited contract SHALL carry a research section stating the spec's fields, that the gate owner approves the objective and the evaluation only, that the spec may evolve while run history is never rewritten and runs stay attributed to the spec version they ran under, that the tracker holds runs and the request holds decisions, that a task closes on its completion condition with negative results as valid outcomes, and that archiving follows completion; the section SHALL be written in the platform's vocabulary.

#### Scenario: Contract read after disposal
- **WHEN** a fresh agent reads the deposited contract in a research repository
- **THEN** it can name the spec's required fields, what the gate covers, how a spec revision treats earlier runs, and when a task archives

#### Scenario: Positive result demanded
- **WHEN** a user asks the deposited contract to require a positive result before a task may close
- **THEN** the contract as deposited states that completion is the completion condition, not success, and the builder records the deviation only if the user reaffirms it

### Requirement: Behavior: The research-task schema is installed through the tool
When the research-task approach is settled on the change workflow tool, the builder SHALL install the schema asset (a research artifact, a hypotheses artifact, a tasks artifact; no specs artifact; changes marked spec-less) as a project-local schema using the tool's own schema commands verified from its help, SHALL record how a change selects the schema, and SHALL fill the platform template lines with the research variant in the second phase.

#### Scenario: Schema installed
- **WHEN** the tool's schema commands exist in the pinned version
- **THEN** the project carries the `research-task` schema at the tool's schema path and the contract names the selection rule

#### Scenario: Schema commands absent
- **WHEN** the pinned tool version has no schema command
- **THEN** the builder places the schema files at the tool's documented schema path by hand, records the version and date, and names the command to verify on the next tool bump
