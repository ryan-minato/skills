# meta/meta-agent-authority Specification

## Purpose
Governs what an agent that loaded the `meta-agent-authority` builder observably does about the approval gate under a specification contract and the vocabulary of the deposited policy.

## Requirements

### Requirement: Behavior: The approval gate precedes review admission and pays H1's price
Under a specification contract, the builder SHALL state that the approval gate precedes review admission on the same change request (combined) or is the specification change request's own review (split), SHALL treat the gate passing on an agent-authored specification as satisfying H1's specification precondition, and SHALL never grant an agent the approval of a specification it wrote.

#### Scenario: H1 offered with an agent-authored specification
- **WHEN** the project has a combined-shape specification contract with a human gate owner and the user asks whether H1 requires the human to write the specification
- **THEN** the builder answers that the gate passing on the agent's specification pays H1's price and that the agent still may not approve its own specification

#### Scenario: Agent asked to approve and ready
- **WHEN** an agent operating at H1 is asked to approve its own specification and mark the pull request ready
- **THEN** the policy the builder deposits makes the agent stop at the gate and hand the approval to the gate owner

### Requirement: Behavior: The deposited policy uses platform verbs
The deposited policy SHALL describe gated actions with the project's platform operations (for example marking a pull request ready, requesting review, approving, merging, releasing) and SHALL define its own gate names in the file.

#### Scenario: Reading the deposited policy
- **WHEN** a clean-context agent reads the deposited policy for a GitHub project
- **THEN** it finds the gated actions named as pull request operations and the gate names defined in the same file
