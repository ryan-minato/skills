## MODIFIED Requirements

### Requirement: Behavior: The approval gate precedes review admission and pays H1's price
Under any contract that defines a gate before review admission — a specification contract's approval gate is the one this repository knows — the builder SHALL state that the gate precedes review admission on the same change request (combined) or is the specification change request's own review (split), SHALL state that the gate passes in the mode the contract records (discussion-closed: the gate owner closes the discussion in conversation and the agent's reconciliation finds nothing open or the open items confirmed; blocking: the gate owner's fixed-wording comment), SHALL treat the gate passing on an agent-authored specification as satisfying H1's specification precondition, and SHALL never grant an agent the passing of a gate on an artifact it wrote.

#### Scenario: H1 offered with an agent-authored specification
- **WHEN** the project has a combined-shape specification contract with a human gate owner and the user asks whether H1 requires the human to write the specification
- **THEN** the builder answers that the gate passing on the agent's specification pays H1's price and that the agent still may not pass its own gate

#### Scenario: Agent asked to approve and ready
- **WHEN** an agent operating at H1 is asked to approve its own specification and mark the pull request ready
- **THEN** the policy the builder deposits makes the agent stop at the gate and hand the decision to the gate owner

#### Scenario: Discussion-closed gate under H1
- **WHEN** the contract records the discussion-closed mode and the gate owner closes the discussion with every thread resolved
- **THEN** the deposited policy lets the agent proceed to design and implementation, and ready still follows implementation and verification

## ADDED Requirements

### Requirement: Behavior: The acceptance-evidence report points at the change request's record and results
Under a specification contract, the deposited report format SHALL have its goal item point at the change request's specification block (the record link) and its tests item at the request's validation section, restating neither.

#### Scenario: Reading the deposited report format
- **WHEN** a clean-context agent reads the deposited policy for a project with a specification contract
- **THEN** it finds the report's goal and tests items pointing at those sections
