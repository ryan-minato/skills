# engineering/brownfield-intelligence Specification

## Purpose
Governs what an agent that loaded the `brownfield-intelligence` skill observably does when it leads work on an existing codebase: identifying the scenario and the depth, bootstrapping the scope, keeping a task tree, finishing independent work before asking, batching human decisions at the dependency frontier, routing each task to the suite member that owns it, and resuming across sessions.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request is broad, multi-step work on an existing, inherited, or legacy codebase whose starting point is unclear — getting a team oriented, preparing the codebase for spec-driven or agent-driven development, or preparing a rewrite — and SHALL not cause it to load for a single focused question about the code, for adopting spec-driven development or configuring a specification tool in a project, or for defining the goals of new software.

#### Scenario: Inherited monolith
- **WHEN** the user says "we just inherited a ten-year-old Rails monolith with almost no docs; where do we even start?"
- **THEN** the skill loads

#### Scenario: Before agents take over
- **WHEN** the user says "before we let coding agents loose on this legacy codebase, get it to a state where they can work safely — figure out what matters and what must not break"
- **THEN** the skill loads

#### Scenario: Focused question (near-miss)
- **WHEN** the user says "what does calculateTax in billing.py return for negative amounts?"
- **THEN** the skill does not load

#### Scenario: Spec tool setup (near-miss)
- **WHEN** the user says "help me set up spec-driven development with OpenSpec for this new project"
- **THEN** the skill does not load

### Requirement: Behavior: Scenario, depth, and scope are set before analysis
The agent SHALL identify the scenario (onboarding, specification, migration, or investigation only) and propose a depth; SHALL write a scope record holding the pinned revision, the goal, the scenario, the depth, the available evidence sources, the permissions (running the system, access to production data or logs, writing to the project), and the initial unknowns; SHALL find out for itself everything the repository can tell it, asking the user only what it cannot discover; and SHALL read no production data, logs, or live system before the user grants that permission.

#### Scenario: Inherited order service
- **WHEN** the user says "we inherited this order service" in a small repository with a README, tests, and a start command
- **THEN** the agent finds the revision, the documents, the tests, and the entry points itself, proposes a scenario and a depth, and asks nothing the repository already answers, such as the language or the framework

#### Scenario: Production data before permission
- **WHEN** the environment holds a connection string for the production database and the user has not yet answered the permission question
- **THEN** the bootstrap reads only the repository, reads no production data or logs, and asks for that permission in its question round

#### Scenario: Ambiguous scenario
- **WHEN** the request fits both onboarding and migration
- **THEN** the agent asks one question with the scenario options and a recommended answer, rather than a questionnaire about facts it could discover

### Requirement: Behavior: The task tree drives the work
The agent SHALL keep a task tree whose nodes each record status, dependencies, evidence, confidence, and open questions; SHALL complete every node not blocked by a human decision before asking; and SHALL limit work on the branches of a pending decision to cheap exploration that informs that decision, marked tentative.

#### Scenario: Independent and blocked nodes
- **WHEN** the tree holds a repository-map node, a domain-vocabulary node, and a node blocked on whether a legacy behavior is intentional
- **THEN** the agent completes the two unblocked nodes before asking about the blocked one

#### Scenario: Decision selects a path
- **WHEN** a pending decision chooses between two migration paths
- **THEN** the agent asks that decision once the independent work is done, and any exploration of the two paths is limited to what informs the decision and labelled tentative

### Requirement: Behavior: Decisions go to the human in frontier batches
The agent SHALL put the mutually independent decisions to the user in one round once the independent work is done, each with its context, evidence, current behavior, recommended default with reasoning, and options with their impact; SHALL present them through the host's structured choice interface when one exists and as a numbered plain-text list otherwise; SHALL record each ruling in the decision records with its authority; and SHALL never record a recommendation as an approval.

#### Scenario: Three independent decisions
- **WHEN** three pending decisions do not depend on one another
- **THEN** the user receives them in one round, each with evidence and a recommended default, rather than as three separate interruptions

#### Scenario: Partial answer
- **WHEN** the user rules on two of the three decisions
- **THEN** the third stays PENDING_DECISION in the decision records, the nodes depending on it stay blocked, and the recommended default is not recorded as approved

### Requirement: Behavior: Work routes to the suite member that owns it
The agent SHALL route analysis, onboarding synthesis, contract specification, and migration work to the suite member that owns each, SHALL not perform that work itself, and SHALL collect the members' results into the shared records.

#### Scenario: Onboarding scenario
- **WHEN** the scenario is onboarding
- **THEN** the evidence nodes are routed to the investigation member, the synthesis node to the onboarding member, and the agent writes no onboarding material itself

### Requirement: Behavior: Independent tasks run in parallel where the host allows
When the host can dispatch clean-context subagents, the agent SHALL dispatch unblocked, mutually independent task nodes in parallel with self-contained read-only briefs and reconcile the results itself, recording conflicting results as CONTRADICTED; when the host cannot, it SHALL run the same briefs sequentially in dependency order with the same output shape; and it SHALL never dispatch a node blocked on a human decision.

#### Scenario: Subagents available
- **WHEN** four unblocked nodes are independent and the host can dispatch subagents
- **THEN** the agent dispatches them in parallel, each brief carrying the scope, the pinned revision, the question, the prohibitions, and the finding format, and it alone writes the reconciled results to the records

#### Scenario: No subagents
- **WHEN** the same four nodes are due and the user states that no subagents may be used
- **THEN** the agent runs them one after another in dependency order and records results in the same format

### Requirement: Behavior: Resume checks drift before reuse
On resuming from an existing workspace, the agent SHALL compare the recorded revision with the current one, mark the findings on paths that changed as stale and re-verify them before relying on them, continue from the task tree, and not re-ask a settled decision unless new evidence contradicts it, in which case it SHALL present that evidence and ask for an explicit revision.

#### Scenario: Code changed since the last session
- **WHEN** the payment module changed after the recorded revision
- **THEN** the agent marks the payment findings stale and re-verifies them before use, keeps the unaffected findings, and continues from the task tree without re-asking settled decisions

### Requirement: Handoff: investigation
When the plan needs evidence and the investigation member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, keep the scope record, task tree, and decision records, mark the evidence nodes blocked, and state that the suite member is missing.

#### Scenario: Handoff offered
- **WHEN** an evidence node is due and the investigation member is not installed
- **THEN** the agent names the investigation role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the evidence nodes are marked blocked with the missing member named, and the scope record, task tree, and decision records are kept

### Requirement: Handoff: onboarding
When the scenario is onboarding and the onboarding member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, deliver the findings as they stand and state that no onboarding material was written.

#### Scenario: Handoff offered
- **WHEN** the onboarding synthesis node is due and the onboarding member is not installed
- **THEN** the agent names the onboarding role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent delivers the findings as they stand and states that no onboarding material was written

### Requirement: Handoff: contract specification
When the scenario is specification and the specification member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, deliver the contract candidates as findings, promote nothing, and state that the suite member is missing.

#### Scenario: Handoff offered
- **WHEN** the classification node is due and the specification member is not installed
- **THEN** the agent names the contract specification role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the contract candidates are delivered as findings, nothing is promoted, and the missing member is named

### Requirement: Handoff: migration
When the scenario is migration and the migration member of the suite is not installed, the agent SHALL route its installation through the installing skill without printing an install command, and SHALL, when the user declines, deliver the findings as they stand, build no baseline, and state that the suite member is missing.

#### Scenario: Handoff offered
- **WHEN** the baseline node is due and the migration member is not installed
- **THEN** the agent names the migration role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the findings are delivered as they stand, no baseline is built, and the missing member is named
