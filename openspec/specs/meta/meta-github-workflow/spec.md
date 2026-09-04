# meta/meta-github-workflow Specification

## Purpose
Governs what an agent that loaded the `meta-github-workflow` builder observably does when it consumes platform-worded deposits, expresses the change request shape and archive mode on GitHub, and ships or designs the archive automation.

## Requirements

### Requirement: Behavior: Platform-worded deposits are implemented, not translated again
The builder SHALL read the workflow deposit at `.agents/knowledge/github-workflow.md` (or the path the entrypoint records), implement the objects it names, and append label semantics, tracking conventions, and mechanics to that same file instead of creating a separate planning file.

#### Scenario: Deposit already present
- **WHEN** the target carries `.agents/knowledge/github-workflow.md` naming its objects
- **THEN** the builder creates no separate planning knowledge file and records its additions in the existing one

### Requirement: Behavior: Take-work follows the change request shape
The generated project skill SHALL, under the combined shape, take a work item that has no specification by committing the change record to a draft pull request first and waiting for the gate owner's recorded approval, and SHALL, under the split shape, escalate a work item whose specification pull request is not merged.

#### Scenario: Combined shape, no specification yet
- **WHEN** the contract records the combined shape and an agent takes a work item with an empty specification field
- **THEN** the generated skill directs it to publish the change record on a draft pull request and wait for the approval comment rather than escalating

### Requirement: Behavior: Templates carry two specification checklist items
The pull request template SHALL carry one item confirming the change record was approved before implementation (or that the request carries the specification only) and one confirming the specification was updated or the change record archived or completed for automated archiving, without changing the security item's wording.

#### Scenario: Checklist check still passes
- **WHEN** a pull request body is built from the adapted template and every item is ticked
- **THEN** the project's checklist check passes

### Requirement: Behavior: A specification-only change request references its work item
Under the split shape, the specification pull request SHALL reference the work item without closing it, and the final implementation pull request SHALL close it.

#### Scenario: Split shape
- **WHEN** the contract records the split shape and the generated skill opens the specification pull request
- **THEN** its body references the work item with a non-closing reference

### Requirement: Behavior: OpenSpec projects get a runnable archive job; other tools get design guidance
Under automated archiving with OpenSpec, the builder SHALL produce a Actions workflow from its asset that runs after merge to the default branch, serialized by a concurrency group, calls the project's archive script (the one the spec-driven development skill ships, copied into the project's scripts directory), fails without retry on a rejected push, and SHALL record the push authorization the automation identity needs as a maintainer action; for Spec-Kit, Kiro, and committed documents it SHALL design the job with the user from the same skeleton and copy no OpenSpec command.

#### Scenario: OpenSpec with automation
- **WHEN** the contract records OpenSpec and automated archiving
- **THEN** the produced Actions workflow declares a concurrency group, calls the project's archive script, and its documentation records the maintainer action

#### Scenario: Spec-Kit with automation
- **WHEN** the contract records Spec-Kit and automated archiving
- **THEN** the builder asks for the completion criterion and post-processing step and produces no OpenSpec command
