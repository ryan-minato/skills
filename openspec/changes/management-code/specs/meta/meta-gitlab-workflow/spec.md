## ADDED Requirements

### Requirement: Behavior: Scripts delivered to the target come from management assets the target owns
The builder SHALL deliver every script it gives the target — the commit check, and the pipeline-log digest and next-version helpers of the durable project skill — from its management assets, never as a copy of a script the builder runs itself; SHALL change only an asset's marked settings; and SHALL tell the user that the target owns the delivered scripts.

#### Scenario: Durable project skill with CI diagnosis
- **WHEN** agents will diagnose GitLab CI in the target and the builder delivers the durable project skill
- **THEN** the pipeline-log digest inside the project skill comes from the management asset, and no delivered file is byte-identical to a file in the builder's `scripts/`

#### Scenario: SemVer not chosen
- **WHEN** the project did not choose SemVer
- **THEN** no next-version helper is delivered

### Requirement: Behavior: Delivered job scripts fail on the failing command
The CI jobs the builder delivers SHALL run their shell lines so that a failing command, including one inside a pipe, fails the job, handling an expected nonzero exit at that command, and SHALL run a command whose failure matters before the condition that tests its result.

#### Scenario: Commit check job
- **WHEN** mechanical commit enforcement is selected and the builder writes the commit-check job
- **THEN** the job runs the delivered commit check under a shell with `errexit` and `pipefail` set, so a failing check or a failing command in a pipe fails the job
