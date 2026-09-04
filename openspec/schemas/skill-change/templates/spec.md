## Purpose
<!-- New domains only: one or two sentences (50+ characters) on what an agent that loaded this skill observably does. Delete for an existing domain. -->

## ADDED Requirements

### Requirement: Trigger: description
<!-- Exactly one per skill, always this name. The skill description SHALL cause the skill to load when ... and SHALL not when ... -->

#### Scenario: <!-- a prompt that loads it -->
- **WHEN** the user says "<!-- prompt -->"
- **THEN** the skill loads

#### Scenario: <!-- a near-miss that must not load it -->
- **WHEN** the user says "<!-- prompt sharing vocabulary -->"
- **THEN** the skill does not load

### Requirement: Behavior: <!-- capability, named freely -->
<!-- The agent SHALL ... -->

#### Scenario: <!-- an outcome task -->
- **WHEN** <!-- the user asks for ... in situation ... -->
- **THEN** <!-- the output shows ... (observable, gradable) -->

#### Scenario: <!-- an edge or refusal -->
- **WHEN** <!-- the situation that must be refused or handled -->
- **THEN** <!-- the observable handling -->

### Requirement: Handoff: <!-- role of the target skill, never its name -->
<!-- The agent SHALL route ... through the installing skill and SHALL ... when the user declines. Delete if the skill pairs with nothing. -->

#### Scenario: Handoff offered
- **WHEN** <!-- the situation that needs the target -->
- **THEN** <!-- the routing through the installing skill -->

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** <!-- the fallback -->

### Requirement: Script: <!-- file name -->
<!-- One per bundled script. The script SHALL ... Delete if the skill bundles no script. -->

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage and exits 0

#### Scenario: Representative run
- **WHEN** <!-- the documented success case -->
- **THEN** <!-- the observable result -->

#### Scenario: Repeated run
- **WHEN** the identical command runs a second time
- **THEN** nothing changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option
