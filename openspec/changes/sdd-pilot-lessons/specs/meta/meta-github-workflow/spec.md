## REMOVED Requirements

### Requirement: Behavior: Take-work follows the change request shape
**Reason**: the platform base carries no paradigm content; the take-work precondition is an extension slot the specification builder fills.
**Migration**: the behavior is required of `meta/meta-spec-workflow`; a project built from an older base keeps its filled steps.

### Requirement: Behavior: Templates carry two specification checklist items
**Reason**: the base's template carries only structural slots; the specification items are the specification builder's insertion.
**Migration**: `meta/meta-spec-workflow` inserts the block and the items; the security item's invariant stays with the base.

### Requirement: Behavior: A specification-only change request references its work item
**Reason**: split-shape behavior is paradigm content.
**Migration**: required of `meta/meta-spec-workflow`.

### Requirement: Behavior: OpenSpec projects get a runnable archive job; other tools get design guidance
**Reason**: the archive workflow asset and the script are the specification builder's.
**Migration**: required of `meta/meta-spec-workflow`, which ships the script and the workflow asset.

## ADDED Requirements

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
