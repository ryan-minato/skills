# engineering/brownfield-onboarding Specification

## Purpose
Governs what an agent that loaded the `brownfield-onboarding` skill observably does when it prepares onboarding material for engineers joining an existing codebase: building a minimum sufficient, trustworthy mental model from investigated evidence, reusing verified documentation, and keeping observed practice, rules, and unknowns apart.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when an engineer new to an existing project needs to become productive in it — an onboarding guide, a newcomer's architecture overview, a glossary of the project's terms, or "where do I change what" — and SHALL not cause it to load for writing agent instruction files such as AGENTS.md, for a README of a newly created package, or for saving one lesson into the project's knowledge.

#### Scenario: Joining next week
- **WHEN** the user says "I start on this team next week — build me an onboarding guide for this codebase so I know where things are and how they fit together"
- **THEN** the skill loads

#### Scenario: Messy docs, new hires
- **WHEN** the user says "our docs are a mess and two engineers join next month; write the material they should read first"
- **THEN** the skill loads

#### Scenario: Agent instructions (near-miss)
- **WHEN** the user says "write an AGENTS.md so coding agents follow our conventions in this repo"
- **THEN** the skill does not load

#### Scenario: New package README (near-miss)
- **WHEN** the user says "write the README for this npm package I created yesterday"
- **THEN** the skill does not load

### Requirement: Behavior: The material is a minimum sufficient mental model
The agent SHALL cover what the system is for, how to run it, the main components, the domain vocabulary, an architecture overview, representative flows, where to change what, known risks and unknowns, and further reading, and SHALL not catalogue every business rule, write tests, or propose refactoring as part of the material.

#### Scenario: Guide for an order service
- **WHEN** the agent writes onboarding material for a small order-service repository
- **THEN** the material has each of the listed parts, each backed by evidence from the repository, and contains no refactoring proposal and no exhaustive rule catalogue

### Requirement: Behavior: Evidence comes first and is not re-derived
The agent SHALL build the material from investigation findings: it SHALL use an existing ledger when one exists and, when none exists, SHALL first gather findings through the investigation role at ORIENT depth for the lenses the material needs; every factual statement in the material SHALL trace to a finding.

#### Scenario: No prior investigation
- **WHEN** the user asks for onboarding material in a repository with no investigation findings
- **THEN** the agent gathers findings at ORIENT depth first and writes material whose statements can each be traced to one of them

#### Scenario: Existing ledger
- **WHEN** the repository's investigation workspace already holds a ledger of findings, one of them UNKNOWN
- **THEN** the material's statements cite those findings, the UNKNOWN finding appears among the material's unknowns, and no statement contradicts the ledger

### Requirement: Behavior: Verified documentation is reused, not duplicated
The agent SHALL spot-check representative claims of the existing documentation; SHALL link and reuse the parts that hold, filling only the gaps; SHALL rebuild a baseline only where the documentation is contradicted or fragmented; and SHALL flag each contradicted claim instead of copying it.

#### Scenario: Partly accurate README
- **WHEN** the README's setup steps work but its statement that payments are retried three times contradicts code that retries five times
- **THEN** the material links the README's setup section instead of copying it, and flags the retry statement as contradicted with both locations

### Requirement: Behavior: Observed practice is not written as a rule, and nothing unverified is presented as working
The agent SHALL describe observed conventions as current practice, never with MUST or SHALL; SHALL mark each unknown together with where to investigate it; and SHALL verify that every path the material names exists and every command it gives runs, marking any command it could not run as unverified with the reason.

#### Scenario: Consistent module layout
- **WHEN** nine of ten modules share one directory layout
- **THEN** the material says the modules currently follow that layout and states no requirement that new modules must

#### Scenario: Command that cannot run here
- **WHEN** the start command needs a database that is not available to the agent
- **THEN** the material shows the command marked unverified with the reason instead of presenting it as working

### Requirement: Handoff: investigation
When no investigation findings exist and the investigation member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, stop before writing any material and state that the suite member is missing and what the material would lack without it.

#### Scenario: Handoff offered
- **WHEN** onboarding material is requested, no findings exist, and the investigation member is not installed
- **THEN** the agent names the investigation role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent writes no onboarding material and states which suite member is missing and what the material would lack
