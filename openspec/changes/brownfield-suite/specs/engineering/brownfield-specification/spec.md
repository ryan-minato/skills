## Purpose
Governs what an agent that loaded the `brownfield-specification` skill observably does when it turns an existing system's observed behavior into normative specification: classifying contract candidates, putting the normative decisions to a human, promoting only approved contracts that carry a verified guardrail, and hardening only approved rules into checks.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request is to decide which behaviors of an existing system are contracts that must be kept, to turn what consumers actually depend on into enforced specification, or to make implicit conventions of an existing codebase explicit and enforced before agents or engineers change it, and SHALL not cause it to load for writing a specification for new behavior, for adopting spec-driven development or configuring a specification tool, or for recording a single convention in agent instructions.

#### Scenario: Real contracts versus accidents
- **WHEN** the user says "which behaviors of our legacy payments API are real contracts we must keep, and which are accidents? I want the real ones written down as specs"
- **THEN** the skill loads

#### Scenario: Consumers depend on events
- **WHEN** the user says "other teams consume our order events — turn what they actually rely on into an enforced contract before agents start changing this service"
- **THEN** the skill loads

#### Scenario: New feature spec (near-miss)
- **WHEN** the user says "write the spec for the new CSV export feature before we build it"
- **THEN** the skill does not load

#### Scenario: Tool setup (near-miss)
- **WHEN** the user says "set up OpenSpec in this repository"
- **THEN** the skill does not load

### Requirement: Behavior: Candidates are classified by the rewrite and consumer tests
The agent SHALL classify each contract candidate as SPECIFICATION, COMPATIBILITY, ARCHITECTURE, IMPLEMENTATION, or UNKNOWN by asking whether a complete reimplementation could change it and who would be affected if it changed, and SHALL classify a candidate as UNKNOWN, not IMPLEMENTATION, when no consumer was found but consumers cannot be ruled out.

#### Scenario: Mixed candidates
- **WHEN** the candidates are "orders are stored in PostgreSQL", "`OrderService` calls `PricingService`", and "a repeated submission with the same idempotency key never creates a second order"
- **THEN** the first two are classified ARCHITECTURE or IMPLEMENTATION and the third is a SPECIFICATION candidate awaiting a decision

#### Scenario: No consumer found
- **WHEN** an event field has no consumer in the repository and the event is published to a shared broker
- **THEN** the field is classified UNKNOWN with the reason, not IMPLEMENTATION

### Requirement: Behavior: A human decides normative status
The agent SHALL put normative decisions to the user in batched rounds, each item stating the evidence, the current behavior, a recommended default with its reasoning, and the options with their impact; SHALL record each ruling with its authority (the user's instruction, an approved requirement, a decision record, a confirmed consumer contract, or a compatibility requirement); SHALL never treat "the code does it" as authority; and SHALL leave an item PENDING_DECISION, unpromoted, until it is ruled on.

#### Scenario: Batched round
- **WHEN** three candidates need rulings that do not depend on one another
- **THEN** the agent puts all three to the user in one round, each with evidence, current behavior, a recommended default, and options with impact

#### Scenario: Item left unanswered
- **WHEN** the user rules on two of the three items and says nothing about the third
- **THEN** the third stays PENDING_DECISION and appears in no normative specification

### Requirement: Behavior: Promotion requires a verified guardrail
The agent SHALL promote an approved contract into normative specification only together with an executable guardrail — a contract test or a check — that passes against the current system and is shown to fail when the contract is deliberately violated, and SHALL keep an approved contract that has no such guardrail in the non-normative baseline, marked as approved but unguarded, with the missing guardrail named.

#### Scenario: Guarded promotion
- **WHEN** the user approves the idempotency-key contract
- **THEN** the specification entry is accompanied by a test that passes on the current code, fails when the duplicate check is temporarily removed, and passes again once it is restored

#### Scenario: Approved but unguarded
- **WHEN** the user approves a contract whose behavior no test can currently exercise
- **THEN** the contract appears in the baseline marked approved but unguarded, names the guardrail that is missing, and does not appear in the normative specification

### Requirement: Behavior: Specifications state contracts, not implementation
The agent SHALL write promoted contracts in the project's existing specification format and location, through the project's own change process when it has one, and SHALL state inputs, outputs, errors, observable side effects, ordering, consistency, idempotency, compatibility, and invariants without naming classes, functions, frameworks, storage engines, or internal call paths unless they are themselves the contract.

#### Scenario: Contract text
- **WHEN** the agent writes the idempotency-key contract
- **THEN** the text states the observable guarantee at the boundary and names no internal class, function, or storage engine

### Requirement: Behavior: Observed conventions become rules only by decision
The agent SHALL distinguish observed conventions, architecture policies, and normative contracts; SHALL not turn an observed pattern into a rule, a check, or an agent instruction without an explicit decision; and, when hardening an approved rule, SHALL prefer a check with a failing exit code over prose and SHALL show each check failing on a violation before relying on it.

#### Scenario: Majority pattern
- **WHEN** nine of ten modules share a directory layout and no document requires it
- **THEN** the agent asks whether the layout should become an architecture policy and writes no rule, check, or instruction for it meanwhile

#### Scenario: Approved rule hardened
- **WHEN** the user approves the rule that the web layer never imports the persistence layer directly
- **THEN** the agent adds a check that fails on such an import, demonstrates the failure on a deliberate violation, and removes the violation

### Requirement: Handoff: investigation
When the contract candidates have not been investigated and the investigation member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, stop before classifying and state that the suite member is missing.

#### Scenario: Handoff offered
- **WHEN** the user asks which behaviors are contracts, no findings exist, and the investigation member is not installed
- **THEN** the agent names the investigation role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent classifies nothing and states which suite member is missing

### Requirement: Handoff: spec-driven development
When the user wants the promoted contracts maintained through a spec-driven workflow the project does not have yet, the agent SHALL offer the spec-driven development role through the installing skill without printing an install command, and SHALL, when the user declines, write the promoted contracts to one contract document beside their guardrails that states it is the source of truth for those boundaries.

#### Scenario: Handoff offered
- **WHEN** the user says future changes to these contracts should go through spec-driven development and the project has no specification workflow
- **THEN** the agent names the spec-driven development role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent writes the promoted contracts to one contract document beside their guardrails, marked as the source of truth for those boundaries
