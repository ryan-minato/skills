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
