## Purpose
Governs what an agent that loaded the `meta-harness-architecture` builder observably does when it writes or deposits a project's management code: the role it is written for, its language, dependencies, and invocation, its error handling, and its shell.

## ADDED Requirements

### Requirement: Behavior: Management code is written for its role and owned by the target
When the build needs a script, hook, or CI step, the builder SHALL write or deposit management code written for that role and owned by the target, never bound to a skill's bundled runtime script — no rule, check, or instruction keeps the two identical, although their content may match where the role needs the same code — and SHALL tell the user that the project owns the code and that later fixes to the skill do not reach it.

#### Scenario: Bundled script available
- **WHEN** a skill used in the build ships a runtime script that does what the target's CI needs
- **THEN** the target receives a script written for the CI role, nothing in the target requires it to stay identical to the skill's script, and the handoff says the project owns it

### Requirement: Behavior: Management code is written in a language the project's community scripts in
The builder SHALL write management code in the project's language when that language's community widely and maturely uses it for scripting, judged by the maturity of that practice and not by whether the language has a build step, and checked against current official documentation for a borderline ecosystem; otherwise in Python or Deno; and in Bash only for simple operations that certainly run in a controlled Linux environment.

#### Scenario: Go project
- **WHEN** the builder adds a check script to a Go project
- **THEN** the script is written in Go and run with `go run`

#### Scenario: C++ project
- **WHEN** the builder adds a check script to a C++ project whose build has no scripting practice of its own
- **THEN** the script is written in Python or Deno, and not in C++ or as a long Bash script

### Requirement: Behavior: Management code dependencies and invocation fit every environment it runs in
The builder SHALL keep a management script self-contained where its ecosystem has no widespread convention for tool dependencies (PEP 723 inline metadata for Python) and follow the convention where one exists; SHALL use the standard library only when it solves the task cleanly, and recommend adding uv to the harness instead of reimplementing a mature library whose standard-library version would need its own tests to be trusted; SHALL invoke a script through `uv run` only when uv exists in every environment the script runs in, and call the interpreter otherwise; and SHALL record the chosen language and invocation in the target's harness knowledge.

#### Scenario: Node project dependency
- **WHEN** a management script in a Node project needs a third-party package
- **THEN** the package is added to the project's `package.json` `devDependencies`, not declared inline in the script

#### Scenario: YAML parsing without uv
- **WHEN** a Python management script must parse YAML and uv is not installed in every environment the script runs in
- **THEN** the builder recommends adding uv to the harness with a PEP 723 header, writes no hand-rolled YAML parser, and writes no `uv run` for an environment that lacks uv

#### Scenario: Invocation recorded
- **WHEN** the builder finishes delivering management scripts
- **THEN** the target's harness knowledge names their language and the command that runs them

### Requirement: Behavior: Management code fails fast at its interfaces and hides no failure
The builder SHALL write management code readable first and failing as early as possible: every interface with an external tool, command, API, or a file another step wrote SHALL be checked for what the code relies on and fail at once with a message naming the interface, the value received, and what to fix; a check's finding SHALL point at the file to fix and print no traceback; no default SHALL stand in for a failure — no fallback on a guaranteed key, swallowed exception, catch-all handler, or handler that only restates the exception — and data already checked at its interface SHALL NOT be checked again inside; values the data contract documents as possibly null SHALL be handled as domain logic; and structural safety, idempotence, and narrow retries on known-transient errors SHALL stay.

#### Scenario: Validator finding
- **WHEN** the builder writes a check that validates a committed file and the file violates it
- **THEN** the check prints a message naming the file and the fix and exits non-zero without a traceback

#### Scenario: External command output
- **WHEN** a management script consumes an external command's JSON output
- **THEN** it checks the command's exit status and the parsed shape where it calls the command, fails naming the command when either is wrong, and reads the fields directly afterwards without `.get(...) or {}` fallbacks

#### Scenario: Documented null
- **WHEN** a script reads a pull request body, which the platform documents as null when the body is empty
- **THEN** it treats null as empty text and adds no guard to fields the platform guarantees

### Requirement: Behavior: Shell in management code fails on the failing command
The builder SHALL write shell steps under `set -euo pipefail` unless a documented exception applies, handling an expected nonzero exit at that command instead of dropping the flags for the whole script; SHALL set `defaults.run.shell: bash` in the GitHub Actions workflows it generates; and SHALL run a command whose failure matters before the condition that tests its result, never inside the condition.

#### Scenario: Generated workflow
- **WHEN** the builder generates a GitHub Actions workflow with `run:` steps
- **THEN** the workflow sets `defaults.run.shell: bash`

#### Scenario: Command inside a condition
- **WHEN** a generated step decides from a command's output whether a check must run
- **THEN** the command runs on its own before the test, so its failure fails the step, and a failed command is never read as "nothing to check"
