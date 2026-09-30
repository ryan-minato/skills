# meta/meta-github-workflow Specification

## Purpose
Governs what an agent that loaded the `meta-github-workflow` builder observably does when it consumes platform-worded deposits and delivers a paradigm-neutral GitHub lifecycle base whose templates and project skill carry registered extension slots for a paradigm builder to fill.

## Requirements

### Requirement: Behavior: Platform-worded deposits are implemented, not translated again
The builder SHALL read the workflow deposit at `.agents/knowledge/github-workflow.md` (or the path the entrypoint records), implement the objects it names, and append label semantics, tracking conventions, and mechanics to that same file instead of creating a separate planning file.

#### Scenario: Deposit already present
- **WHEN** the target carries `.agents/knowledge/github-workflow.md` naming its objects
- **THEN** the builder creates no separate planning knowledge file and records its additions in the existing one

### Requirement: Behavior: Templates and the project skill carry paradigm-neutral extension slots
The delivered pull request template, issue forms, and project skill SHALL contain no paradigm-specific line, field, or step, SHALL keep the headings, step positions, and field ids the builder's slot list names so that a paradigm builder can insert its lines by structure alone, SHALL contain no placeholder or anchor comment once delivered, and the builder SHALL hand off to the builder whose description claims the contract the entrypoint points to.

#### Scenario: Specification contract present at build time
- **WHEN** the target carries a specification contract and the builder delivers the base
- **THEN** the delivered template and project skill contain no specification line or step, every slot heading and field id is present, and the hand-off names the paradigm shaping as the next step

#### Scenario: No paradigm contract
- **WHEN** the target carries no paradigm contract
- **THEN** the delivered files contain no placeholder and no anchor comment, and the slot structures are still present

#### Scenario: Security line survives a slot fill
- **WHEN** a paradigm builder later inserts items into the checklist slot
- **THEN** the security item's wording is unchanged and the checklist check passes

### Requirement: Behavior: A required check is named in the ruleset only after it has run on the default branch
The builder SHALL sequence a new required check as workflow first, live and green on the default branch, then the ruleset entry, and SHALL say that a check the platform has never observed blocks every pull request.

#### Scenario: New check named
- **WHEN** a new workflow job is to become a required check
- **THEN** the builder merges the workflow and confirms a run on the default branch before editing the ruleset

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
The workflows the builder delivers SHALL hide no unexpected failure behind a fallback and SHALL mark every deliberate deferral where it is written, the aggregator gate being the designed point that decides on the needed jobs' results and fails whenever one failed or was cancelled; the taxonomy check SHALL declare its YAML dependency in PEP 723 inline metadata and run through `uv run`, the builder recommending uv for the target's harness where a developer environment lacks it and presenting dependency locking as the user's decision, defaulted by the job's risk.

#### Scenario: Workflows delivered
- **WHEN** the builder delivers the checks, commit-check, taxonomy, and tag-check workflows
- **THEN** no delivered step reads an unexpected failure as success, every deliberate deferral is marked where it is written, and the gate step, run as its workflow runs it with a needs result containing a failed job, exits non-zero

#### Scenario: Taxonomy check without local uv
- **WHEN** the target's developers do not have uv installed
- **THEN** the builder recommends adding uv to the target's harness for the taxonomy check instead of bundling a YAML parser, says that locking is optional for the read-only taxonomy job, and writes no lock the user did not choose
