## ADDED Requirements

### Requirement: Behavior: The hand-off order runs the paradigm builder's second phase after the platform base
The builder SHALL hand off in this order — the specification workflow builder's contract phase when spec-driven intent was recorded, then agent authority, then the platform lifecycle builder that delivers the paradigm-neutral base, then the specification workflow builder again to shape that base — and the deposited workflow file SHALL point at other contracts (authority, any paradigm contract) generically without restating them.

#### Scenario: Hand-off order names the second phase
- **WHEN** spec-driven intent was recorded and the builder finishes its deposit
- **THEN** its hand-off lists the contract phase before authority and the platform builder, and the shaping phase after them

#### Scenario: Deposited file without a paradigm contract
- **WHEN** no paradigm contract exists at deposit time
- **THEN** the deposited workflow file points at the authority policy and records that no paradigm contract exists yet, with no paradigm-specific section
