## ADDED Requirements

### Requirement: Behavior: Scripts delivered to the target come from management assets the target owns
The builder SHALL deliver every script it gives the target — the commit check, the taxonomy check, the label sync committed beside `labels.json`, and the run-log digest, next-version, and project-field helpers of the durable project skill — from its management assets rather than from the scripts the builder runs itself, with no rule, check, or instruction binding a delivered script to one of those (an asset may still match one where the role needs the same code); SHALL change only an asset's marked settings; and SHALL tell the user that the target owns the delivered scripts.

#### Scenario: Label taxonomy committed
- **WHEN** the approved taxonomy is committed to the target as `labels.json`
- **THEN** the label sync script delivered beside it comes from the builder's management asset, and nothing delivered to the target requires it to stay identical to a file in the builder's `scripts/`

#### Scenario: Durable project skill with CI diagnosis
- **WHEN** the builder delivers the durable project skill with the Actions-diagnosis branch selected
- **THEN** the run-log digest inside the project skill comes from the management asset, and the project skill runs it from the project's own path

#### Scenario: SemVer not chosen
- **WHEN** the project did not choose SemVer and did not opt into Projects
- **THEN** no next-version or project-field helper is delivered

### Requirement: Behavior: Delivered workflows and the taxonomy check follow the management-code rules
The workflows the builder delivers SHALL let no step pass after a command it depends on failed — the aggregator gate fails whenever a needed job failed or was cancelled — whether the step fails at that command or when it ends; the taxonomy check SHALL declare its YAML dependency in PEP 723 inline metadata and run through `uv run`, the builder recommending uv for the target's harness where a developer environment lacks it and presenting dependency locking as the user's decision, defaulted by the job's risk.

#### Scenario: Workflows delivered
- **WHEN** the builder delivers the checks, commit-check, taxonomy, and tag-check workflows
- **THEN** no delivered step passes after a command it depends on failed, and the gate step, run as its workflow runs it with a needs result containing a failed job, exits non-zero

#### Scenario: Taxonomy check without local uv
- **WHEN** the target's developers do not have uv installed
- **THEN** the builder recommends adding uv to the target's harness for the taxonomy check instead of bundling a YAML parser, says that locking is optional for the read-only taxonomy job, and writes no lock the user did not choose
