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

### Requirement: Behavior: Management code follows the fail-fast philosophy and hides no failure
The builder SHALL apply fail-fast as a design philosophy that guides judgment, not as a fixed procedure: management code is readable first; a failure that matters is never hidden — no default, fallback on a guaranteed key, swallowed exception, catch-all handler, or handler that only restates the exception turns it into a pass or a wrong answer — and nothing is built on a result the code knows is bad; where the code relies on an external tool, command, API, or a file another step wrote, it checks what it relies on and fails with a message naming the interface, the value received, and what to fix; a check's finding points at the file to fix and prints no traceback; when the failure surfaces — at the failing call, or after collecting findings at the end of the script or step — is chosen for readability and for the person who fixes it; values the data contract documents as possibly null are handled as domain logic; and structural safety, idempotence, and narrow retries on known-transient errors stay.

#### Scenario: Validator finding
- **WHEN** the builder writes a check that validates a committed file and the file violates it
- **THEN** the check prints a message naming the file and the fix and exits non-zero without a traceback

#### Scenario: Findings collected before failing
- **WHEN** the user asks for a check that reports every violating file in one run
- **THEN** the check collects the findings, reports each with its file, and fails once at the end, and the builder does not restructure it to stop at the first finding

#### Scenario: External command output
- **WHEN** a management script consumes an external command's JSON output
- **THEN** a wrong exit status or an unexpected shape ends the script failed with a message naming the command, and no default stands in for the data it did not get

#### Scenario: Documented null
- **WHEN** a script reads a pull request body, which the platform documents as null when the body is empty
- **THEN** it treats null as empty text and adds no guard to fields the platform guarantees

### Requirement: Behavior: Shell in management code does not lose a failure
The builder SHALL keep the shell it writes from losing a failure that matters — a command's failure is not read as a false condition, swallowed by `|| true`, or hidden behind a pipe — so that the step or job ends failed, whether at the failing command or when the step ends, and SHALL treat strict mode (`set -euo pipefail`) and a workflow-level `defaults.run.shell: bash` as tools it may use where they make that simplest, not as rules every step must follow.

#### Scenario: Command inside a condition
- **WHEN** a generated step decides from a command's output whether a check must run
- **THEN** a failure of that command fails the step instead of being read as "nothing to check"

#### Scenario: Status checked when the step ends
- **WHEN** a generated step pipes the output of a command whose failure matters into another command
- **THEN** the step ends failed when that command failed, either through `pipefail` or through a status the step checks before it ends, and neither form is treated as a defect
