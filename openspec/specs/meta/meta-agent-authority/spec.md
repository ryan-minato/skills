# meta/meta-agent-authority Specification

## Purpose
Governs what an agent that loaded the `meta-agent-authority` builder observably does about the approval gate under a specification contract and the vocabulary of the deposited policy.

## Requirements

### Requirement: Behavior: The deposited policy uses platform verbs
The deposited policy SHALL describe gated actions with the project's platform operations (for example marking a pull request ready, requesting review, approving, merging, releasing) and SHALL define its own gate names in the file.

#### Scenario: Reading the deposited policy
- **WHEN** a clean-context agent reads the deposited policy for a GitHub project
- **THEN** it finds the gated actions named as pull request operations and the gate names defined in the same file

### Requirement: Behavior: The acceptance-evidence report points at the change request's record and results
Under a specification contract, the deposited report format SHALL have its goal item point at the change request's specification block (the record link) and its tests item at the request's validation section, restating neither.

#### Scenario: Reading the deposited report format
- **WHEN** a clean-context agent reads the deposited policy for a project with a specification contract
- **THEN** it finds the report's goal and tests items pointing at those sections

### Requirement: Behavior: The contract's two gates precede review admission and integration, and pay H1's price
Under any contract that defines gates before the policy's own — a specification contract's are the ones this repository knows — the builder SHALL state that there are two: the package gate, which precedes review admission on the same change request (combined) or is the specification change request's own review (split), and the freeze gate, which precedes integration and is exercised on the finished implementation after the request is marked ready. The builder SHALL state that each passes in the mode the contract records — conversational: the gate owner closes the deliberation in conversation and the agent's reconciliation finds nothing open or the open items confirmed; recorded: the mark the contract names on the version being approved — SHALL treat the package gate passing on an agent-authored specification as satisfying H1's specification precondition, and SHALL never grant an agent the passing of a gate on an artifact it wrote, so an agent that may mark a request ready still waits for the package gate first and never freezes a record before the freeze gate closes.

#### Scenario: H1 offered with an agent-authored specification
- **WHEN** the project has a combined-shape specification contract with a human gate owner and the user asks whether H1 requires the human to write the specification
- **THEN** the builder answers that the package gate passing on the agent's specification pays H1's price and that the agent still may not pass its own gate

#### Scenario: Agent asked to approve and ready
- **WHEN** an agent operating at H1 is asked to approve its own specification and mark the change request ready
- **THEN** the policy the builder deposits makes the agent stop at the package gate and hand the decision to the gate owner

#### Scenario: A gate closes under H1
- **WHEN** the contract records the conversational mode and the gate owner closes the package deliberation with every thread resolved
- **THEN** the deposited policy lets the agent proceed to the task list and implementation, and ready still follows implementation and verification

#### Scenario: Asked to freeze before the second gate
- **WHEN** an agent operating at H1 has marked the request ready and is asked to archive the record before the gate owner has closed the deliberation on the finished implementation
- **THEN** the deposited policy makes the agent decline, say the freeze follows that gate, and leave the specification check red until it does

### Requirement: Behavior: The archive executor is read as the freeze before approval
Where a specification contract is present, the builder SHALL read its archive executor as the answer to who archives inside the change request once the deliberation on the finished implementation closes — the implementer by default, and a maintainer on a contributor's branch for a request from a fork — SHALL treat that archive as the freeze the approval applies to, and SHALL never present archiving as a job that runs after the merge or as an automation that pushes an archive commit.

#### Scenario: Asked what precedes approval
- **WHEN** the project's contract records an archive executor and the user asks what the agent must finish before the maintainer approves
- **THEN** the builder answers that the deliberation closes, the implementer archives inside the request, and approval applies to that frozen version

#### Scenario: Fork request
- **WHEN** the change request comes from a fork
- **THEN** the builder names a maintainer as the executor on the contributor's branch and says no automation can push there

#### Scenario: No contract
- **WHEN** the project has no specification contract
- **THEN** the builder says the archive executor is unset, that its absence blocks nothing, and still deposits the authority policy
