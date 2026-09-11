## Purpose
Governs what an agent that loaded the `meta-python-defaults` builder observably does when it settles a Python project's documentation, testing, and toolchain defaults, including the deviations a tensor-heavy project takes.

## ADDED Requirements

### Requirement: Trigger: description
The builder description SHALL open with the disposable-builder marker and SHALL cause the builder to load during a harness build when a Python project lacks settled docstring, comment, test, dependency, lint, format, type-check, task, hook, or documentation conventions, and SHALL not cause it to load for writing the project's production code or for migrating a settled toolchain.

#### Scenario: Unsettled toolchain
- **WHEN** the user says "the harness is coming together; we still have no decision on linting, tests, docstrings, or hooks for this Python service"
- **THEN** the builder loads

#### Scenario: Production code (near-miss)
- **WHEN** the user says "write the retry decorator for our HTTP client"
- **THEN** the builder does not load

### Requirement: Behavior: Tensor-heavy projects take the experiment deviations
When the project's core is tensor code — training, fine-tuning, or inference on a deep-learning framework — the builder SHALL not make static type checking a gate over that code and SHALL offer it only for configuration and control-plane modules, SHALL set docstring conventions that carry shape, dtype, device, layout, and mask semantics, SHALL keep GPU-only tests in an explicit suite that fails without hardware and out of the default suite, SHALL keep heavy validation manual, and SHALL configure git hooks that run the formatter and linter only because commits are experiment snapshots; the deposit SHALL record each deviation with its reason.

#### Scenario: Type checker proposed as a gate
- **WHEN** the user asks for a type-check gate on a PyTorch training project
- **THEN** the builder declines the gate over the model code, offers it for the configuration schema, and records the reason

#### Scenario: Hook runs tests
- **WHEN** the project's hook runs the test suite and the project trains models
- **THEN** the builder proposes formatter-and-linter-only hooks, moves tests to the task runner, and records the snapshot-commit reason

#### Scenario: Ordinary Python service
- **WHEN** the project is a web service with no tensor code
- **THEN** the builder applies its default baseline, including static checking where the project benefits, and the tensor deviations do not appear
