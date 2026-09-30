## MODIFIED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request names an Agent Skill, a SKILL.md, or an agent instruction package, or asks to create, review, or repair one, including one that never gets picked up or never triggers, and SHALL not cause it to load for application code, documentation written for people, or a mechanism other than a skill that never triggers.

#### Scenario: Skill authoring request
- **WHEN** the user says "Write a SKILL.md so our agents follow the release checklist every time"
- **THEN** the skill loads

#### Scenario: Misbehaving skill, indirect phrasing
- **WHEN** the user says "the instruction package I gave my agent for changelog entries never gets picked up — fix it"
- **THEN** the skill loads

#### Scenario: Misbehaving skill, direct phrasing
- **WHEN** the user says "my changelog-entries skill never triggers when I ask for a changelog entry — its SKILL.md is under .agents/skills — fix it"
- **THEN** the skill loads

#### Scenario: Human documentation (near-miss)
- **WHEN** the user says "write a README that explains how to run the release checklist"
- **THEN** the skill does not load

#### Scenario: Human skills (near-miss)
- **WHEN** the user says "which skills should a junior engineer build first?"
- **THEN** the skill does not load

#### Scenario: Non-skill trigger (near-miss)
- **WHEN** the user says "the pre-commit hook I added never triggers when I commit — fix it"
- **THEN** the skill does not load
