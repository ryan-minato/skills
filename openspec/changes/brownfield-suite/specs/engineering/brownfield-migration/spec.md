## Purpose
Governs what an agent that loaded the `brownfield-migration` skill observably does when an existing system is rewritten, re-platformed, or re-architected: building a compatibility baseline of boundary behavior, pinning it with classified characterization tests, agreeing an equivalence envelope with a human, and planning verification of the new system without silent fixes or accidental fossilization.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when an existing system or component is to be rewritten in another language, moved to another architecture or platform, or replaced wholesale while the behavior its consumers rely on must survive — including "what must stay the same" and "how do we prove the new one behaves like the old one" — and SHALL not cause it to load for small-step refactoring inside one codebase, for routine dependency upgrades, or for performance tuning.

#### Scenario: Language rewrite
- **WHEN** the user says "we're rewriting this Python order service in Go — what has to stay the same, and how do we prove the new one matches?"
- **THEN** the skill loads

#### Scenario: Extracting a service
- **WHEN** the user says "billing is moving out of the monolith into its own service next quarter; how do we make sure clients don't notice?"
- **THEN** the skill loads

#### Scenario: Refactoring one function (near-miss)
- **WHEN** the user says "clean up this 400-line function without changing what it does"
- **THEN** the skill does not load

#### Scenario: Performance work (near-miss)
- **WHEN** the user says "this endpoint takes three seconds; make it faster"
- **THEN** the skill does not load

### Requirement: Behavior: The compatibility baseline separates contract, compatibility, and pending behavior
The agent SHALL record, for each boundary behavior in scope, the current behavior, its evidence, the consumers found, and its normative status, and SHALL present every behavior without a decision as PENDING_DECISION with the options PROMOTE_TO_SPEC, PRESERVE_TEMPORARILY, and INTENTIONALLY_CHANGE, a recommended default, and the impact of each option.

#### Scenario: Error that looks accidental
- **WHEN** the legacy service returns HTTP 500 for a malformed legacy order ID
- **THEN** the baseline records the current behavior as 500 with status PENDING_DECISION, notes that external clients may depend on it, and offers the three options with a recommended default, and it does not state that the specification requires a 500

### Requirement: Behavior: No silent fix and no accidental fossilization
The agent SHALL not plan or implement a difference from baseline behavior without a recorded INTENTIONALLY_CHANGE decision, and SHALL not declare a baseline behavior a permanent requirement without a recorded PROMOTE_TO_SPEC decision; a behavior that looks like a bug and has no decision SHALL stay in the baseline as pending-decision and be put to the user.

#### Scenario: Suspected bug without a decision
- **WHEN** the legacy code allows a refund from the `partially_refunded` state, no document mentions it, and no decision exists
- **THEN** the agent keeps the transition in the baseline as pending-decision, pins it with a tagged characterization test, and lists it for the user, and it neither drops it from the new system's plan nor writes it into a normative specification

#### Scenario: User orders a change
- **WHEN** the user says "the new service should return 400 for malformed IDs"
- **THEN** the agent records an INTENTIONALLY_CHANGE decision with the user as its authority and the affected consumers noted, and moves the behavior from the equivalence set to the planned changes

### Requirement: Behavior: Characterization tests pin boundary behavior with a class tag
The agent SHALL write characterization tests at the system's boundaries — requests and responses, errors, persisted state, emitted events, and external calls — each tagged normative, compatibility, or pending-decision; SHALL pin behavior that looks like a bug and tag it rather than correct it; SHALL control nondeterminism (clock, randomness, generated identifiers, ordering) or scrub volatile fields rather than loosening assertions; and SHALL rely on the suite only after it passes twice in a row against the existing system.

#### Scenario: Suite for an order API
- **WHEN** the agent builds characterization tests for a small order API
- **THEN** the tests exercise the HTTP boundary, include the malformed-ID 500 case tagged pending-decision, carry a tag on every test, and pass in two consecutive runs against the existing service

#### Scenario: Timestamp in the response
- **WHEN** each response carries a creation timestamp
- **THEN** the test controls the clock or scrubs that field and still asserts the rest of the body, instead of dropping the body assertion

### Requirement: Behavior: The equivalence envelope compares boundaries, not internals
The agent SHALL record, for each boundary behavior, whether it must be strictly identical, semantically equivalent under a stated rule, allowed to differ, intentionally changed, or still unknown; SHALL exclude internal calls, class structure, module names, internal data structures, and algorithms unless they are themselves contracts; and SHALL obtain the user's approval of the envelope before it is used as acceptance for the new system.

#### Scenario: Envelope for the order API
- **WHEN** the agent drafts the envelope for the order API rewrite
- **THEN** every entry names a boundary behavior and one of the five categories, no entry names an internal function or class, and the envelope is marked awaiting approval until the user approves it

### Requirement: Behavior: Verification uses explicit tolerances that never widen silently
The agent SHALL plan how the new system is checked against the envelope — the characterization suite run against the new system, differential or shadow comparison of responses, and comparison of persisted state and emitted events — with every tolerance stated with its reason, and SHALL not widen a tolerance or reclassify a difference to make a failing comparison pass without a recorded decision.

#### Scenario: Differential run finds a difference
- **WHEN** a differential run shows the new service orders the items of a list response differently and the envelope marks that response strictly identical
- **THEN** the agent reports the difference as a failure and asks whether ordering may differ, and it does not loosen the comparison on its own

### Requirement: Handoff: investigation
When the boundary behavior has not been investigated and the investigation member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, stop before building the baseline and state that the suite member is missing.

#### Scenario: Handoff offered
- **WHEN** a rewrite is planned, no findings exist, and the investigation member is not installed
- **THEN** the agent names the investigation role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent builds no baseline and states which suite member is missing

### Requirement: Handoff: contract specification
When the user rules PROMOTE_TO_SPEC on a behavior and the specification member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, keep the behavior in the baseline marked approved with promotion pending and write no normative specification.

#### Scenario: Handoff offered
- **WHEN** the user rules PROMOTE_TO_SPEC and the specification member is not installed
- **THEN** the agent names the contract specification role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the baseline marks the behavior approved with promotion pending, and no normative specification is written

### Requirement: Handoff: incremental refactoring
When the user decides to restructure the existing code in place in small behavior-preserving steps instead of replacing it, the agent SHALL offer the incremental refactoring role through the installing skill without printing an install command, and SHALL, when the user declines, keep the baseline and the characterization suite as the safety net and leave the step discipline to the user.

#### Scenario: Handoff offered
- **WHEN** the user decides to evolve the legacy code in place rather than rewrite it
- **THEN** the agent names the incremental refactoring role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent hands over the baseline and the characterization suite as the safety net and states that the step discipline is left to the user
