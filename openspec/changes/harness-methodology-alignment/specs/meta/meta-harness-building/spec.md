## Purpose
Governs what an agent that loaded the `meta-harness-building` entry builder observably does when it learns a repository, settles the harness requirement with the user, and rates and plans the harness layers before building them through the skills at hand.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the user asks to set up, build, improve, audit, or repair a repository's agent harness — its AGENTS.md or CLAUDE.md, knowledge files, project skills, checks, CI, permissions — or reports that agents keep missing conventions and the harness may have drifted; and SHALL not cause it to load for an isolated edit to one file, for the practice of a single layer, or for studying another repository's harness.

#### Scenario: Set up a repository for agents
- **WHEN** the user says "set up this repository so coding agents can work in it safely"
- **THEN** the skill loads

#### Scenario: Drifted harness
- **WHEN** the user says "our agents keep ignoring the commit rules and the AGENTS.md is probably stale — sort it out"
- **THEN** the skill loads

#### Scenario: Single lint rule (near-miss)
- **WHEN** the user says "add a ruff rule that forbids print statements"
- **THEN** the skill does not load

#### Scenario: Study of another repository (near-miss)
- **WHEN** the user says "look at how github.com/example/platform sets up its agents and report what is enforced"
- **THEN** the skill does not load

### Requirement: Behavior: The requirement round settles the five variables and the threat-model facts
When the agent settles the requirement with the user, it SHALL first record from inspection what the repository shows (deploy target, history age and author count, sandbox definition, unattended workflows) and SHALL then ask only for what it cannot show: the five variables that set each layer's thickness (error cost, maintenance horizon, environment isolation, agent autonomy, team size), the externally writable input sources agents will read and whether a human reviews results before they leave, which irreversible or outward actions are pre-authorized, and which control-plane files are hard read-only.

#### Scenario: Questions after inspection
- **WHEN** the agent has read a repository with a dev container, a three-author history, and no deploy configuration
- **THEN** its questions name the isolation and team size it inferred as facts to confirm, and ask for error cost, maintenance horizon, agent autonomy, the external inputs agents will read, the human-review point, and the pre-authorized irreversible actions

#### Scenario: Variable the repository shows
- **WHEN** the repository has a committed dev container with scoped tokens
- **THEN** the agent does not ask the user what isolation the environment provides

### Requirement: Behavior: The plan rates six layers with evidence
When the agent designs and presents the harness plan, it SHALL rate six layers — environment and sandbox, knowledge and routing, hard gates, authorization boundary, specification and completion criteria, entropy management — each omitted, light, medium, or thick, with the variable that sets it and, for an omitted layer, the trigger that would build it later, SHALL start from the exploratory or engineering preset and record each per-layer deviation, and SHALL refuse to assign one maturity level to the whole project.

#### Scenario: Plan table
- **WHEN** the agent presents the plan for a long-lived solo library installed by others
- **THEN** the plan's thickness table has exactly six layer rows, each with a rating and the variable behind it, rates knowledge and routing and authorization lighter than the engineering preset because team size and autonomy are low, and keeps hard gates thick because of error cost

#### Scenario: One level requested
- **WHEN** the user asks the agent to classify the project as "exploratory" and skip the per-layer table
- **THEN** the agent keeps the six rows, uses the exploratory preset as the starting column, and records the layers that deviate from it
