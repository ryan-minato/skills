## Purpose
Governs what an agent that loaded the `meta-python-defaults` builder observably records for a Python project's management scripts: how their dependencies are declared, how they are invoked, how they handle errors, and how a CI job's script dependencies are locked.

## ADDED Requirements

### Requirement: Behavior: Python management scripts declare dependencies inline and run with what every environment has
The builder SHALL record that a Python management script with third-party dependencies declares them in PEP 723 inline metadata and runs through `uv run` only when uv exists in every environment the script runs in; that a script whose task the standard library solves cleanly stays standard-library-only when uv is not guaranteed and runs with the interpreter; and that a mature library is never reimplemented to avoid a dependency, uv being recommended for the harness instead.

#### Scenario: Dependency with uv everywhere
- **WHEN** a project whose dev container and CI both install uv needs a management script that parses YAML
- **THEN** the recorded convention gives the script a PEP 723 header declaring the YAML library and runs it with `uv run`

#### Scenario: No uv in CI
- **WHEN** the project's CI does not install uv and a management script only reads JSON and runs git
- **THEN** the recorded convention keeps the script standard-library-only and runs it with the Python interpreter, not with `uv run`

### Requirement: Behavior: Python management scripts check their interfaces and hide no failure
The builder SHALL record an error-handling idiom for Python management scripts, presented as the fail-fast philosophy rather than a fixed procedure, in which each subprocess, HTTP, and file-read interface the script relies on is checked and a mismatch exits with a message naming the interface and the value received; data checked there is read directly afterwards; and a bare or blind `except`, a `try`/`except`/`pass`, or a handler that only restates the exception is an exception that needs a stated reason, never a default habit.

#### Scenario: Error-handling convention recorded
- **WHEN** the builder records the conventions for the project's management scripts
- **THEN** the convention names the interface checks that exit with a message, and treats blind handlers, silent handlers, and handlers that only restate the exception as exceptions that need a stated reason

### Requirement: Behavior: Locking a CI script's dependencies is the user's decision, defaulted by the job's risk
When a PEP 723 script with third-party dependencies runs in a CI job, the builder SHALL present locking its dependencies — `uv lock --script`, or an `exclude-newer` cutoff — as a recommendation whose default follows the job's actual risk: recommended for a job with a privileged trigger, a writable token, or secrets, and not required for a read-only job without secrets; and SHALL apply no lock the user did not choose.

#### Scenario: Privileged job
- **WHEN** the script runs in a `pull_request_target` job whose token can write
- **THEN** the builder recommends locking as the default, names both options, and waits for the user's decision before writing either

#### Scenario: Read-only check
- **WHEN** the script runs in a `pull_request` job with a read-only token and no secrets
- **THEN** the builder says locking is optional for that job and applies nothing the user did not choose
