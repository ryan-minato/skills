## Purpose
Governs what an agent that loaded the `experiment-code-conventions` skill observably does when it writes, reviews, or restructures machine-learning experiment code: how it abstracts, which dependencies it prefers, how it shapes the training loop and the configuration surface, what it tests and lints, and how it trades readability against hot-path performance.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request concerns the shape of training or experiment code — sharing versus duplicating code, a Trainer versus an explicit loop, vendoring a research repository, what a configuration file may contain, tests, type checking, docstrings, or comments for tensor code, or a readability-versus-performance trade-off in a training path — and SHALL not cause it to load for general application code with no training or tensor concern, or for a refactoring request that names no machine-learning code.

#### Scenario: Trainer or loop
- **WHEN** the user says "our two training scripts are 80% the same — should I pull them into a shared Trainer class?"
- **THEN** the skill loads

#### Scenario: Configuration doing too much
- **WHEN** the user says "the YAML now names optimizer classes by import path and has an `if` for the scheduler — is that fine?"
- **THEN** the skill loads

#### Scenario: Web handler cleanup (near-miss)
- **WHEN** the user says "simplify this Flask request handler and add type hints"
- **THEN** the skill does not load

#### Scenario: Generic refactoring (near-miss)
- **WHEN** the user says "refactor this module safely — it has no tests yet"
- **THEN** the skill does not load

### Requirement: Behavior: Abstraction follows semantic coupling, not similarity
The agent SHALL share an implementation only when two places must stay logically consistent and will change together, SHALL tolerate duplication between code that merely looks alike today and may diverge with the research, SHALL state the rule as "abstract the stable mechanism, duplicate the unstable policy", and SHALL explain once when the user cites repetition alone as the reason and then follow the user's reaffirmed decision.

#### Scenario: Shared preprocessing
- **WHEN** train and evaluation scripts each contain the same preprocessing and a drift would corrupt the comparison
- **THEN** the agent extracts the preprocessing into one shared implementation and says why these two must stay consistent

#### Scenario: Similar loops, different research goals
- **WHEN** two experiment scripts have similar training loops that serve different hypotheses
- **THEN** the agent keeps them separate, names the divergence risk, and does not introduce a shared base class

#### Scenario: Repetition cited as the reason
- **WHEN** the user asks to deduplicate two scripts because they repeat code
- **THEN** the agent asks whether the copies must stay consistent, explains the coupling rule once, and follows the user's decision if they reaffirm it

### Requirement: Behavior: The loop is explicit and dependencies are mature or vendored with their origin
The agent SHALL prefer an explicit training loop with a maintained acceleration library over a framework Trainer, SHALL accept a structured Trainer only when the training semantics are settled, are not a research variable, and are understood by the team, SHALL prefer dependencies maintained by an organization with a record and a test suite, and SHALL treat an individual's or a lab's unmaintained research repository as something to vendor — copying the needed part with its origin commit and license — in replication mode with minimal semantic difference or in innovation mode keeping only the research semantics.

#### Scenario: New training project
- **WHEN** the user starts a training script and asks whether to use a Trainer
- **THEN** the agent writes an explicit loop with the acceleration library's device, precision, and accumulation handling, and names gradient accumulation, loss normalization, zeroing, scheduler stepping, precision boundaries, and evaluation timing as the semantics the loop keeps visible

#### Scenario: Paper repository
- **WHEN** the user wants to depend on a paper's repository that has had no maintainer activity for a year
- **THEN** the agent vendors the needed implementation with its origin commit and license, names the mode (replication or innovation), and does not add the repository as a runtime dependency

### Requirement: Behavior: The configuration surface holds values a run may choose and never becomes a language
The agent SHALL keep configuration to the choices the project intends to expose to a run — hyperparameters, dataset, architecture, algorithm, evaluation, runtime and resource choices — SHALL reject configuration that constructs arbitrary objects, composes through registries or class paths, carries control flow, or inherits deeply, SHALL default an unsettled project to a typed schema with YAML files, command-line overrides, and a resolved dump saved with the run, and SHALL keep an existing composition framework the project already uses while forbidding object instantiation from it.

#### Scenario: Class paths in configuration
- **WHEN** a configuration file selects the optimizer by a `_target_` import path
- **THEN** the agent replaces it with a named choice among the optimizers the code supports and moves construction into code

#### Scenario: Existing composition framework
- **WHEN** the project already uses a configuration composition framework and the user asks whether to switch
- **THEN** the agent keeps it, records the two rules (no object instantiation from configuration; a pinned output directory), and does not migrate

### Requirement: Behavior: Tests protect behavior contracts and the default suite stays light
The agent SHALL test the project's own integration assumptions — tensor shapes, mask semantics, padding, label alignment, reduction semantics, custom layers — and not third-party libraries, SHALL keep the default suite light and CPU-compatible, SHALL keep GPU-only tests as a separate suite that runs only by explicit command and fails when the hardware is absent, SHALL keep expensive validation manual, SHALL keep git hooks free of tests because commits are experiment snapshots, and SHALL not use coverage as a quality target.

#### Scenario: GPU test without a GPU
- **WHEN** the user asks to make the GPU tests skip automatically when no GPU is present
- **THEN** the agent keeps them failing under the explicit GPU suite, excludes them from the default suite, and explains that a silent skip hides a broken test

#### Scenario: Hook that runs tests
- **WHEN** the project's pre-commit hook runs the test suite
- **THEN** the agent limits the hook to formatting and lint, moves the tests to the task runner, and names the snapshot-commit latency as the reason

### Requirement: Behavior: Style favors readable tensor code without a global type gate
The agent SHALL apply a near-default linter and formatter configuration, SHALL not make global static type checking a quality gate over tensor-heavy code, SHALL write annotations that aid reading and document shape, dtype, device, layout, and mask conventions in docstrings or nearby comments, SHALL comment at block level to explain mathematical intent, non-obvious invariants, numerical reasons, and performance rationale, and SHALL keep hot-path performance while recovering understandability through local encapsulation, comments, tests, a benchmark, or a clear slower reference implementation.

#### Scenario: Type checker proposed for model code
- **WHEN** the user asks to make the type checker pass on the model and kernel code as a CI gate
- **THEN** the agent declines the global gate, keeps annotations where they aid reading, adds shape and dtype documentation, and offers static checking for configuration and control-plane modules only

#### Scenario: Unfusing for readability
- **WHEN** a reviewer asks to split a fused hot-path computation into readable steps
- **THEN** the agent keeps the fused implementation, adds a block comment with the rationale, and adds a slower reference implementation with a test that compares the two

### Requirement: Handoff: training instrumentation
When the user asks what the loop should log or how its metrics should be judged, the agent SHALL offer the instrumentation role through the installing skill without printing an install command, and SHALL, when the user declines, keep a single logging seam and a single stage-trace seam in the loop and leave the metric design to the user.

#### Scenario: Handoff offered
- **WHEN** the user asks which metrics the new training loop should emit
- **THEN** the agent names the instrumentation role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent leaves one logging seam and one stage-trace seam in the loop and records that metric design was not covered
