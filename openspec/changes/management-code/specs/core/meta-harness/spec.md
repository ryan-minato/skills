## Purpose
Governs what an agent that loaded the `meta-harness` methodology observably does with a project's management code — the checks, scripts, hooks, and CI shell that never ship in the product — when it designs, audits, or changes a harness.

## ADDED Requirements

### Requirement: Behavior: Management code belongs to the project
The agent SHALL treat code that never ships in the product — quality checks, environment preparation, CI and administration scripts, git hooks, inline workflow shell, and scripts inside project skills — as the project's own management code: written for its role, readable first, guided by the fail-fast philosophy — a failure it cannot handle surfaces as a failure with a message that names what to fix, never as a pass or a wrong answer — and never bound to a skill's runtime script: no rule, check, or instruction requires it to stay identical to one, although its content may match one where the role needs the same code.

#### Scenario: Skill script offered as the project's check
- **WHEN** the user asks the agent to add a CI check to a project, and an installed skill's bundled script already performs that check
- **THEN** the agent gives the project a script of its own, written for the CI role, and adds no rule, check, or instruction that keeps it identical to the skill's script

#### Scenario: Audit finds a bound script
- **WHEN** an audit of a project's harness finds a project script kept byte-identical to a skill's bundled script by a sync rule or a check
- **THEN** the agent reports the binding rule or check as a finding and proposes removing it so the script becomes the project's own, and does not report the matching content alone as a defect
