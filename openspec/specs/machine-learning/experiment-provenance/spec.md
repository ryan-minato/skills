# machine-learning/experiment-provenance Specification

## Purpose
Governs what an agent that loaded the `experiment-provenance` skill observably does when it establishes, records, or judges the identity of a machine-learning run: the source snapshot, the resolved configuration, the environment, the inputs, and the tracker that holds them.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request concerns whether a training or evaluation run can be reproduced, what a run must record, which run produced an artifact, or wiring or choosing an experiment tracker, and SHALL not cause it to load for organizing a series of hypotheses into a research task, for scaffolding a new project, or for build provenance of software artifacts outside machine learning.

#### Scenario: Reproducibility question
- **WHEN** the user says "our fine-tuning results differ between last week's run and today's and I can't tell what changed — what should every run record so this stops happening?"
- **THEN** the skill loads

#### Scenario: Tracker wiring
- **WHEN** the user says "add experiment tracking to train.py so each run's config and metrics are kept somewhere"
- **THEN** the skill loads

#### Scenario: Research task organization (near-miss)
- **WHEN** the user says "we want to try four tokenizer variants over the next two weeks — how should we organize the experiments and the pull request?"
- **THEN** the skill does not load

#### Scenario: Software build provenance (near-miss)
- **WHEN** the user says "make our CI record which git commit built each Docker image and attach an SBOM"
- **THEN** the skill does not load

### Requirement: Behavior: Run identity is composed of four immutable parts plus a run id
When asked what a run must record or whether a run is reproducible, the agent SHALL name the source snapshot (the executed commit), the resolved configuration, the environment identity (a container image digest or the dependency lock digest, with the host facts a performance claim needs), and the identities of every input that affects the result, SHALL mint a run id distinct from the commit so one snapshot may produce many runs, and SHALL refuse a branch name, a tag such as `latest`, a Dockerfile, or a working-directory path as an identity.

#### Scenario: Run record contents
- **WHEN** the user asks what their run record should contain
- **THEN** the agent lists the four parts and the run id, states that the run id is not the commit, and names for each part the immutable form it takes

#### Scenario: Mutable references offered as identity
- **WHEN** the user's record names the dataset as `main` and the image as `latest`
- **THEN** the agent resolves each to an immutable identity (a dataset revision or checksum, an image digest), records the resolved value, and explains why the mutable name is not an identity

#### Scenario: Performance claim without host facts
- **WHEN** the user wants to record a run whose result is a throughput improvement and the record carries only the image digest
- **THEN** the agent adds the GPU model, driver, and runtime versions to the record and states that the digest alone does not explain a performance result

### Requirement: Behavior: The resolved configuration is saved whole and is immutable after start
The agent SHALL save the configuration actually in effect after every source is merged (defaults, project and experiment files, command-line overrides, search suggestions, runtime-derived values) with the run, SHALL not accept the command line, a partial override, or the raw YAML as that record, SHALL treat scheduler-driven state such as a changing learning rate as training state rather than configuration mutation, SHALL use "hyperparameter" only for values that are hyperparameters and call other searched values search variables, and SHALL refuse to rewrite the recorded configuration of a completed run.

#### Scenario: Config assembled from several sources
- **WHEN** a run's configuration comes from defaults, a YAML file, and command-line overrides
- **THEN** the agent writes the merged, resolved configuration to the run's record and the tracker, and does not record only the command line

#### Scenario: Correcting a past run's record
- **WHEN** the user asks to edit yesterday's recorded configuration to what they meant to run
- **THEN** the agent declines to overwrite it, records a correction note or a new run that carries the intended configuration, and keeps the original record intact

#### Scenario: Naming a searched value
- **WHEN** the user calls the dataset choice and the GPU count "hyperparameters" in a sweep definition
- **THEN** the agent names them search variables (data configuration and resource configuration) and reserves "hyperparameter" for values such as learning rate and weight decay

### Requirement: Behavior: The source snapshot is committed before the run and stays reachable
Before launching a run from a tree with uncommitted changes, the agent SHALL make a snapshot commit on the experiment branch or worktree so the recorded commit equals the executed source, SHALL record the commit with the run, and, when history will be squashed or a branch deleted, SHALL name the retention rule that keeps every referenced snapshot reachable (a tag or a kept ref) rather than recording a commit that garbage collection can drop.

#### Scenario: Launch from a dirty tree
- **WHEN** the user asks the agent to launch a run and the working tree has uncommitted edits
- **THEN** the agent commits a snapshot on the experiment branch first and records that commit as the run's source

#### Scenario: Squash planned
- **WHEN** the user plans to squash the experiment branch into one commit on the main branch
- **THEN** the agent names the rule that keeps the referenced snapshot commits reachable and records it in the project's conventions

### Requirement: Behavior: The tracker is selected by precedence and holds the manifest without secrets
The agent SHALL keep a working tracker the project already uses, SHALL otherwise select the platform's experiment tracking when the project is hosted on a platform that provides it, and SHALL otherwise default to a lightweight open tracker; SHALL emit the run manifest to the tracker as parameters and tags and to the run's output directory; and SHALL never put credentials, presigned URLs, raw samples, or prompts into the tracker or the manifest.

#### Scenario: Existing tracker
- **WHEN** the project already logs to a working tracker and the user asks the agent to record provenance
- **THEN** the agent adds the manifest fields to that tracker and does not propose a migration

#### Scenario: Prompts proposed as run metadata
- **WHEN** the user wants the tracker to store the prompts of each evaluation sample for later debugging
- **THEN** the agent records sample identifiers or hashes instead and directs raw samples to a governed artifact store with access control

### Requirement: Handoff: research task organization
When the request turns from one run's record to organizing several hypotheses, runs, and a pull or merge request into one research task, the agent SHALL offer the research-task role through the installing skill without printing an install command, and SHALL, when the user declines, answer the run-record question alone and state that task organization was left out.

#### Scenario: Handoff offered
- **WHEN** the user, after settling the run record, asks how to organize the next month of experiments into a pull request
- **THEN** the agent names the research-task role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent answers only the run-record question and records that the research-task organization was not covered
