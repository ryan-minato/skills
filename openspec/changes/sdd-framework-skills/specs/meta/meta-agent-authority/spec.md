## ADDED Requirements

### Requirement: Behavior: The archive executor is read as a step inside the change request
Where a specification contract is present, the builder SHALL read its archive executor as the answer to who archives inside the change request before it is marked ready — the agent by hand, or the automation the project's framework skill installs — and SHALL never present archiving as a job that runs after the merge.

#### Scenario: Contract names the label automation
- **WHEN** the project's contract records that the framework skill's label automation archives, and the user asks what the agent must finish before marking the request ready
- **THEN** the builder answers that the change is archived inside the request by that automation, and does not offer an after-merge job

#### Scenario: No contract
- **WHEN** the project has no specification contract
- **THEN** the builder says the archive executor is unset, that its absence blocks nothing, and still deposits the authority policy
