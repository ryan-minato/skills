# meta/meta-workflow-design Specification

## Purpose
Governs what an agent that loaded the `meta-workflow-design` builder observably does when it designs in the platform-neutral model, treats the hosting platform as a required fact, and deposits the workflow file in that platform's vocabulary.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a project's way of tracking, planning, or accepting work is being designed or redesigned, and SHALL not cause it to load for expressing an existing design on a platform or for a one-off issue or pull request operation.

#### Scenario: Workflow design request
- **WHEN** the user says "we're growing from one maintainer to a team of four; how should we track and plan work?"
- **THEN** the skill loads

#### Scenario: Platform expression request
- **WHEN** the user says "create the GitHub labels and issue forms for the workflow we already decided"
- **THEN** the skill does not load

### Requirement: Behavior: Design summary in the neutral model, deposit in platform vocabulary
The builder SHALL present its design summary in the management model's vocabulary and SHALL deposit the workflow file at `.agents/knowledge/<platform>-workflow.md` in the evidenced platform's object vocabulary, translated through the sibling platform builder's semantic mapping, with omitted semantics recorded as platform objects deliberately not used.

#### Scenario: GitHub-hosted project
- **WHEN** the builder runs for a project whose remote is GitHub and the user approves a design with an objective boundary and no timebox
- **THEN** the summary shown to the user says "Objective Boundary", the deposited file says "milestone", lists iterations and boards under objects not used with their triggers, and a search of the deposited file for the model nouns (objective boundary, timebox, planning surface, tracked work, change request, draft change, source marker, work hierarchy) finds nothing

### Requirement: Behavior: The platform is a required fact
When the inspection cannot evidence the hosting platform, the builder SHALL ask which platform the project uses in its first questioning round and SHALL not deposit the workflow file until it is answered.

#### Scenario: Local repository without a remote
- **WHEN** the repository has no remote, no CI directory, and no platform templates
- **THEN** the builder asks which platform the project will use and writes no workflow file until the user answers

### Requirement: Behavior: A specification-only draft is work in progress
The management model SHALL treat a draft change request whose first content is a specification as work that has begun, never as a placeholder.

#### Scenario: Draft carrying only the change record
- **WHEN** the user asks whether a draft pull request containing only a proposal and specification violates the draft rule
- **THEN** the builder answers that it is work in progress under the model and points at the specification contract for the shape
