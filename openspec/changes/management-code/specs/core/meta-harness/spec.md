## Purpose
Governs what an agent that loaded the `meta-harness` methodology observably does with a project's management code — the checks, scripts, hooks, and CI shell that never ship in the product — when it designs, audits, or changes a harness.

## ADDED Requirements

### Requirement: Behavior: Management code belongs to the project
The agent SHALL treat code that never ships in the product — quality checks, environment preparation, CI and administration scripts, git hooks, inline workflow shell, and scripts inside project skills — as the project's own management code: written for its role, readable first, failing as early as possible with a message that names what to fix, and never a copy of a skill's runtime script that the project is bound to keep identical.

#### Scenario: Skill script offered as the project's check
- **WHEN** the user asks the agent to add a CI check to a project, and an installed skill's bundled script already performs that check
- **THEN** the agent writes a script of the project's own for the CI role, and neither copies the skill's script nor adds a rule that keeps a project file identical to it

#### Scenario: Audit finds a copied script
- **WHEN** an audit of a project's harness finds a project script kept byte-identical to a skill's bundled script by a sync rule or a check
- **THEN** the agent reports the coupling as a finding and proposes making the script the project's own, written for its role
