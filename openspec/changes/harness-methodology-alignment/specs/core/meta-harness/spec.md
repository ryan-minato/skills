## ADDED Requirements

### Requirement: Behavior: Thickness is rated per layer on six layers and five variables
When the agent designs, calibrates, audits, or explains a harness, it SHALL rate six layers separately — environment and sandbox, knowledge and routing, hard gates, authorization boundary, specification and completion criteria, entropy management — each as omitted, light, medium, or thick, SHALL justify each rating by the five variables that set it (error cost, maintenance horizon, environment isolation, agent autonomy, team size), naming which variable pushes the layer away from the project's overall profile, and SHALL refuse to assign one maturity level or one task-type mode to the whole project.

#### Scenario: Mixed project
- **WHEN** the user asks how thick the harness of a research repository should be, and the repository holds an exploratory notebooks area and a data pipeline that production reports depend on
- **THEN** the agent's output rates each of the six layers separately, rates hard gates and entropy management thicker for the pipeline than for the notebooks, and names error cost and maintenance horizon as the variables that split them

#### Scenario: One level requested
- **WHEN** the user asks the agent to "just pick a maturity level for the whole project"
- **THEN** the agent declines to assign one level, explains that the variables differ by layer, and delivers six per-layer ratings with one reason each

### Requirement: Behavior: Soft boundaries are split by loading behavior and process-critical skills are routed explicitly
The agent SHALL place every soft boundary by when it must load — always, by task type, by location, by action, or by description match — SHALL keep every rule that must hold on every task in the root entrypoint's always-applicable part, SHALL let a subdirectory entry file or a framework-specific carrier (path rule, hook injection) hold only local supplements that never contradict the root, and SHALL route a process-critical skill from the entrypoint by situation rather than rely on description matching, which it reserves for auxiliary skills.

#### Scenario: Compliance review skill
- **WHEN** the user asks where to wire a project skill that must run a privacy review before every commit
- **THEN** the agent's output adds an explicit entrypoint route ("before committing, run the <skill>") and states that description matching alone is not enough for a skill whose miss has consequences

#### Scenario: Safety rule in a subdirectory file
- **WHEN** the agent audits a harness whose only "never commit credentials" rule sits in a subdirectory's `CLAUDE.md`
- **THEN** the agent reports it as a rule that may never load, moves it to the root entrypoint's always-applicable part, and leaves the subdirectory file with local supplements only

### Requirement: Behavior: Hard boundaries are placed where they are enforced and route errors to their fix
The agent SHALL place defect prevention in CI and git hooks, environment and secret protection in the sandbox and scoped tokens, and framework permission configuration as fool-proofing only; SHALL prefer a tool's native quiet or compact output flags over custom log parsers and accept native noise when no flag exists; and SHALL make every custom check it designs fail with a message that names what to change, the alternative, or the document to read.

#### Scenario: Custom check without routing
- **WHEN** the agent reviews a project's custom pre-commit check whose failure prints only "forbidden import"
- **THEN** the agent's output rewrites the message to name the allowed alternative or the document to read, and does not propose a log-parsing wrapper around the project's test runner

#### Scenario: Framework deny list offered as the security boundary
- **WHEN** the user proposes a framework command deny list as the project's protection against an agent leaking a deployment credential
- **THEN** the agent's output states that an agent with code execution can bypass framework configuration, keeps the deny list as fool-proofing, and places the protection in the sandbox and in the token's scope

### Requirement: Behavior: Isolation is decided by the threat model
The agent SHALL decide whether an agent role needs a strictly isolated, credential-stripped instance by three conditions — the role holds sensitive assets or high privileges, reads externally writable input, and can act outward or change state; the input source is open to anyone (issues, comments, search results, arbitrary pages) rather than reputation-guarded or the trusted repository; and no human reviews results before they leave — requiring isolation when all three hold and ordinary protection otherwise, and SHALL treat model injection resistance only as a modifier for the middle tier, confirmed per model, never as a substitute for isolation.

#### Scenario: Unattended triage of public issues
- **WHEN** the user plans an unattended agent that reads newly opened public issues and holds a token that can push and comment
- **THEN** the agent's output requires a separate restricted instance without the sensitive token or direct write to the main branch, and names the three conditions it found

#### Scenario: Attended reading of official documentation
- **WHEN** the user asks whether an agent working with a developer in the loop may read a tool's official documentation to answer a question
- **THEN** the agent's output places this in ordinary protection, naming the reputation-guarded source and the human in the loop, and does not require isolation

### Requirement: Behavior: The authorization boundary is designed around reversibility and batched decisions
The agent SHALL separate what an agent can do (sandbox and tokens) from what it may do, SHALL move approvals to the plan (results and bounds, never a step list) and to the end (a batched list of pending decisions), SHALL classify actions by reversibility and outward reach so reversible internal actions proceed and irreversible or outward ones are pre-authorized in the specification (or, without one, the approved plan or the policy file) or queued, SHALL direct a blocked agent to continue work that does not depend on the decision, and SHALL write permission lists as explicit allows for known-safe commands, explicit denies for known-dangerous ones, no ask entries without a stated reason, and a policy that forbids the agent from relaxing it.

#### Scenario: Per-command approval requested as security
- **WHEN** the user asks for every shell command to require approval so the agent "cannot do anything dangerous"
- **THEN** the agent's output explains that per-command approval fatigues into no review and is not a security mechanism, proposes pre-authorization in the plan plus a pending-decisions list at the end, and keeps the security boundary in the sandbox and token scope

#### Scenario: Pending decisions format
- **WHEN** the user asks what an agent should do when a planned change needs a decision the plan did not cover
- **THEN** the agent's output says to finish every part that does not depend on it and to record the item with where it is stuck, the options, the recommended option with its reason, and which completed work each option affects

### Requirement: Behavior: Runtime experience is non-interactive with commit checkpoints
The agent SHALL have the harness disable pagers, interactive prompts, and watch modes through environment or command flags, SHALL provide a seconds-scale scoped verification command beside the full check, and SHALL set checkpoint rules — a checkpoint is a commit, never a stash or a framework-only rewind; a task's draft commits are squashed into one atomic commit when the task ends, non-interactively; draft commits never leave the machine; a secret scan runs on every commit; "no draft commits on the branch" is part of the task's done criteria.

#### Scenario: Runtime rules for a new harness
- **WHEN** the user asks what runtime conventions the harness of a long-running agent session needs
- **THEN** the agent's output names non-interactive environment settings, a scoped fast check beside the full one, and the checkpoint rules including the non-interactive squash and the per-commit secret scan

#### Scenario: Stash proposed as checkpoint
- **WHEN** the user proposes `git stash` as the agent's way to save state before a risky edit
- **THEN** the agent's output rejects stash for agents (unnamed, positional, conflict-prone, invisible in the log) and prescribes a commit

### Requirement: Behavior: Harness drift lands somewhere and the harness has checks of its own
The agent SHALL have the harness require that a discovered mismatch between documents, scripts, and code is handled in one of three ways — corrected in place in the same change, queued in the pending-decisions list, or recorded as a tracked item with a fixed label — never left only in conversation; and SHALL, for a long-lived project, give the harness its own deterministic checks in two tiers — reference integrity (paths exist, documented commands exist in the task runner, mirrored files match) and structural compliance (skill layout, strict specification validation, entrypoint length budget) — whose failures name the file and the fix.

#### Scenario: Mismatch found mid-task
- **WHEN** the user asks what an agent should do on finding that the entrypoint documents a command the task runner no longer has
- **THEN** the agent's output gives the three landing places, chooses in-place correction for a small, certain mismatch, and rejects reporting it only in the conversation

#### Scenario: Checks for the harness itself
- **WHEN** the user asks how a long-lived repository keeps its agent documents from rotting
- **THEN** the agent's output proposes the reference-integrity tier first, the structural tier where the project has skills or specifications, and failure messages that name the file to fix
