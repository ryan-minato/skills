## ADDED Requirements

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
