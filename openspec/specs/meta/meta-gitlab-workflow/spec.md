# meta/meta-gitlab-workflow Specification

## Purpose
Governs what an agent that loaded the `meta-gitlab-workflow` builder observably does when it consumes platform-worded deposits and delivers a paradigm-neutral GitLab lifecycle base whose templates and project skill carry registered extension slots for a paradigm builder to fill.

## Requirements

### Requirement: Behavior: Platform-worded deposits are implemented, not translated again
The builder SHALL read the workflow deposit at `.agents/knowledge/gitlab-workflow.md` (or the path the entrypoint records), implement the objects it names, and append label semantics, tracking conventions, and mechanics to that same file instead of creating a separate planning file.

#### Scenario: Deposit already present
- **WHEN** the target carries `.agents/knowledge/gitlab-workflow.md` naming its objects
- **THEN** the builder creates no separate planning knowledge file and records its additions in the existing one

### Requirement: Behavior: Templates and the project skill carry paradigm-neutral extension slots
The delivered merge request template, work-item templates, and project skill SHALL contain no paradigm-specific line, field, or step, SHALL keep the headings, step positions, and field ids the builder's slot list names so that a paradigm builder can insert its lines by structure alone, SHALL contain no placeholder or anchor comment once delivered, and the builder SHALL hand off to the builder whose description claims the contract the entrypoint points to.

#### Scenario: Specification contract present at build time
- **WHEN** the target carries a specification contract and the builder delivers the base
- **THEN** the delivered template and project skill contain no specification line or step, every slot heading and field id is present, and the hand-off names the paradigm shaping as the next step

#### Scenario: No paradigm contract
- **WHEN** the target carries no paradigm contract
- **THEN** the delivered files contain no placeholder and no anchor comment, and the slot structures are still present

#### Scenario: Security line survives a slot fill
- **WHEN** a paradigm builder later inserts items into the checklist slot
- **THEN** the security item's wording is unchanged and the checklist check passes

### Requirement: Behavior: The protected branch baseline includes thread resolution and pre-existing pipeline jobs
The builder SHALL propose, as the whole protected-branch baseline, merging only through a merge request, "pipelines must succeed" enabled only after the referenced jobs exist and pass, and "all threads must be resolved", and SHALL let the user subtract from it with a recorded reason.

#### Scenario: Baseline proposed
- **WHEN** the builder reaches protected-branch settings for a project with a pipeline
- **THEN** it proposes the three items together, orders the pipeline-success rule after the jobs have run, and records any subtraction with its reason

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
