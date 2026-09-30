## ADDED Requirements

### Requirement: Behavior: Scripts delivered to the target come from management assets the target owns
The builder SHALL deliver every script it gives the target — the commit check, and the pipeline-log digest and next-version helpers of the durable project skill — from its management assets rather than from the scripts the builder runs itself, with no rule, check, or instruction binding a delivered script to one of those (an asset may still match one where the role needs the same code); SHALL change only an asset's marked settings; and SHALL tell the user that the target owns the delivered scripts.

#### Scenario: Durable project skill with CI diagnosis
- **WHEN** agents will diagnose GitLab CI in the target and the builder delivers the durable project skill
- **THEN** the pipeline-log digest inside the project skill comes from the management asset, and nothing delivered to the target requires it to stay identical to a file in the builder's `scripts/`

#### Scenario: SemVer not chosen
- **WHEN** the project did not choose SemVer
- **THEN** no next-version helper is delivered

### Requirement: Behavior: Delivered job scripts hide no unexpected failure
The CI jobs the builder delivers SHALL hide no unexpected failure — a failed command whose output is piped on, or whose result a condition tests, is not read as success — and SHALL mark a failure the design tolerates or defers explicitly, such as `allow_failure` on an advisory job.

#### Scenario: Commit check job
- **WHEN** mechanical commit enforcement is selected and the builder writes the commit-check job
- **THEN** a failing commit check fails the job, and a failure of a command whose output the job pipes on is not read as success
