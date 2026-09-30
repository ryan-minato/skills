## ADDED Requirements

### Requirement: Behavior: The label-sync workflow is offered whenever a taxonomy is committed
The builder SHALL, whenever the plan commits a `labels.json`, put the delivery of a label-sync workflow to the user as a numbered frontier question with a recommendation to deliver it, on every repository shape including one with no outside contributors, and SHALL deliver no label-sync workflow when the plan commits no `labels.json` or the user declines it, recording the manual sync path in the delivered knowledge on a decline.

#### Scenario: Taxonomy planned on a solo repository
- **WHEN** the builder presents the design frontier for a personal-account repository with no outside contributors whose plan commits a `labels.json`
- **THEN** the frontier holds a numbered question offering the label-sync workflow, its recommendation is to deliver it, and the question says that the workflow applies each merged change to `labels.json` and never deletes a label

#### Scenario: No taxonomy committed
- **WHEN** the approved plan keeps GitHub's default labels and commits no `labels.json`
- **THEN** the frontier holds no label-sync question and no label-sync workflow is delivered

#### Scenario: Workflow declined
- **WHEN** the plan commits a `labels.json` and the user declines the offered label-sync workflow
- **THEN** no `.github/workflows/labels-sync.yml` is delivered, and the delivered knowledge records the manual sync path: dry-run the delivered sync script, review the plan, and apply it with explicit authorization

### Requirement: Behavior: A weekly label sync is scheduled only for a sole-source taxonomy with a named owner
The builder SHALL include a weekly schedule in the delivered label-sync workflow only when the user confirms that `labels.json` is the sole source of the repository's labels and names the schedule's owner, SHALL record that owner in the platform-settings register, and SHALL say that the schedule reverts label edits made in the web interface.

#### Scenario: Sole source confirmed
- **WHEN** the user accepts the workflow, confirms that `labels.json` is the sole source of labels, and names an owner for the schedule
- **THEN** the delivered workflow carries one `schedule` trigger whose cron fires once a week, the platform-settings register's Labels row names that owner, and the builder says that label edits made in the web interface are reverted on the next scheduled run

#### Scenario: Labels also edited in the web interface
- **WHEN** the user accepts the workflow but says that labels are also edited in the web interface, or names no owner for a schedule
- **THEN** the delivered workflow has no `schedule` trigger, and the builder says that a push-triggered run still applies the whole file and so reverts a web edit to any label the file defines

### Requirement: Behavior: The delivered label-sync workflow applies the committed taxonomy on the default branch
The label-sync workflow the builder delivers SHALL come from its management asset as `.github/workflows/labels-sync.yml` with the job `labels / sync`; SHALL run the delivered `scripts/sync_labels.py` with `--apply` on a push to the default branch that touches the labels file, the sync script, or the workflow, and on manual dispatch; SHALL run only on the default branch with job permissions of `issues: write` and `contents: read` alone; and SHALL be recorded in the target's checks knowledge with its healthy-run shape and never as a required check.

#### Scenario: Workflow delivered
- **WHEN** the builder delivers step 4 for an approved taxonomy with the workflow accepted, in a repository whose local approvals are granted and which permits no remote writes
- **THEN** `.github/workflows/labels-sync.yml` exists with the job name `labels / sync`; its `on:` holds `push` to the default branch with paths for `labels.json`, `scripts/sync_labels.py`, and the workflow file, and `workflow_dispatch`; it declares `permissions: {}` at the top and job permissions of exactly `contents: read` and `issues: write`; its job carries an `if:` restricting it to the default branch's ref; it contains no `--prune`; `grep -n '{{[A-Z]'` over it prints nothing; and the checks knowledge lists `labels / sync` with its healthy-run shape and not among the required checks

#### Scenario: Dispatch from another branch
- **WHEN** the delivered workflow is dispatched from a ref other than the default branch
- **THEN** the job's `if:` condition is false for that ref, the job is skipped, and no label changes

### Requirement: Behavior: The label-sync workflow never deletes a label
The delivered label-sync workflow SHALL never pass `--prune` and SHALL report each repository label absent from `labels.json` as a warning in a run that stays green, and the builder SHALL keep a deletion a manual, authorized step even when the user asks the workflow to prune.

#### Scenario: Label on the repository but not in the file
- **WHEN** the delivered workflow's apply and report steps run as the workflow runs them, against a stub `gh` that records its calls and lists every label of `labels.json`, one of them with a drifted color, plus two labels absent from the file
- **THEN** the stub records one label edit and no label deletion, the report prints one `::warning::` line naming each of the two absent labels, a summary table is written to `GITHUB_STEP_SUMMARY`, and both steps exit 0

#### Scenario: Asked to prune automatically
- **WHEN**, during step 4, the user asks that the workflow also delete the labels missing from `labels.json`
- **THEN** the delivered workflow still carries no `--prune`, the builder says that a deletion strips the label from every issue and pull request carrying it, and the delivered knowledge records the manual path: list the issues and pull requests carrying the label, get explicit authorization, then run the sync script with `--apply --prune`

### Requirement: Behavior: The label-sync workflow reaches the default branch only after the authorized build-time apply
The builder SHALL keep the build-time dry run, review, authorized apply, and readback as the first application of the approved taxonomy; SHALL deliver the label-sync workflow onto the default branch only after that apply was authorized; SHALL tell the user, before the workflow lands, that each later merged change to `labels.json` is applied without a further prompt and that nothing is deleted; and SHALL present the workflow's first default-branch run as a readback expected to be all-skip.

#### Scenario: First run after the build
- **WHEN** the taxonomy was applied during the build with authorization and read back, and the workflow is delivered
- **THEN** the handoff says that the workflow's first run on the default branch, triggered by the merge or by a dispatch, is expected to create and update nothing, that a create or update in it is drift to investigate, that each later merged change to `labels.json` is applied without a further prompt, and that the workflow never deletes a label

#### Scenario: Build-time apply not authorized
- **WHEN** the user withholds authorization to apply the labels during the build
- **THEN** no `.github/workflows/labels-sync.yml` is among the delivered files, the builder says that the workflow's first run would apply the taxonomy, and the handoff records the workflow as pending that authorization instead of presenting it as inert

## MODIFIED Requirements

### Requirement: Behavior: Delivered workflows and the taxonomy check follow the management-code rules
The workflows the builder delivers SHALL hide no unexpected failure behind a fallback and SHALL mark every deliberate deferral where it is written, the aggregator gate being the designed point that decides on the needed jobs' results and fails whenever one failed or was cancelled; the taxonomy check SHALL declare its YAML dependency in PEP 723 inline metadata and run through `uv run`, the builder recommending uv for the target's harness where a developer environment lacks it and presenting dependency locking as the user's decision, defaulted by the job's risk.

#### Scenario: Workflows delivered
- **WHEN** the builder delivers the checks, commit-check, taxonomy, tag-check, and label-sync workflows
- **THEN** no delivered step reads an unexpected failure as success, every deliberate deferral is marked where it is written, the label-sync report that keeps a run with absent labels green among them; the gate step, run as its workflow runs it with a needs result containing a failed job, exits non-zero; and the label-sync apply step, run as its workflow runs it with a stub `gh` that exits 1, exits non-zero

#### Scenario: Taxonomy check without local uv
- **WHEN** the target's developers do not have uv installed
- **THEN** the builder recommends adding uv to the target's harness for the taxonomy check instead of bundling a YAML parser, says that locking is optional for the read-only taxonomy job, and writes no lock the user did not choose
