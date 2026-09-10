# core/great-skill-writing Specification

## Purpose
Governs what an agent that loaded the `great-skill-writing` skill observably does: which requests load it, and the contract of its bundled linter `scripts/lint_skill.py`.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request names an Agent Skill, a SKILL.md, or an agent instruction package, or asks to create, review, or repair one, and SHALL not cause it to load for application code or documentation written for people.

#### Scenario: Skill authoring request
- **WHEN** the user says "Write a SKILL.md so our agents follow the release checklist every time"
- **THEN** the skill loads

#### Scenario: Human documentation (near-miss)
- **WHEN** the user says "write a README that explains how to run the release checklist"
- **THEN** the skill does not load

#### Scenario: Human skills (near-miss)
- **WHEN** the user says "which skills should a junior engineer build first?"
- **THEN** the skill does not load

### Requirement: Script: lint_skill.py
The bundled script SHALL lint a skill directory or its SKILL.md and exit 0 when it finds no error, 1 when it finds at least one error or the skill path does not exist, and 2 on bad arguments; SHALL import PyYAML only when it parses a SKILL.md, so `--help`, unknown options, and a missing skill path complete without the dependency; and when PyYAML is absent at parse time SHALL exit 2 with a diagnostic naming the dependency and how to obtain it, a case its `--help` documents.

#### Scenario: Help
- **WHEN** the script runs with `--help` under an interpreter that cannot import `yaml`
- **THEN** it prints usage that names the exit codes, including the missing-PyYAML case, and exits 0

#### Scenario: Representative run
- **WHEN** the script runs through `uv run` with `--skill` naming a spec-compliant skill directory
- **THEN** it prints the OK line (or warnings only) to stdout and exits 0, and with `--json` prints a JSON array

#### Scenario: Repeated run
- **WHEN** the identical command runs a second time
- **THEN** the output is identical and nothing on disk changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option under an interpreter that cannot import `yaml`
- **THEN** it exits 2 and prints a diagnostic naming the option, with no traceback

#### Scenario: Missing skill path
- **WHEN** `--skill` names a SKILL.md that does not exist, under an interpreter that cannot import `yaml`
- **THEN** it exits 1 with a diagnostic naming the path, with no traceback

#### Scenario: Missing dependency
- **WHEN** `--skill` names an existing skill and the interpreter cannot import `yaml`
- **THEN** it exits 2 and prints to stderr a diagnostic that names PyYAML, says the frontmatter cannot be parsed without it, and tells the user to run the script with `uv run` or install `pyyaml>=6.0,<7`, with no traceback
