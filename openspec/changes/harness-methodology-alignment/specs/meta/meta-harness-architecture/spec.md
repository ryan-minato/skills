## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when a harness build reaches a layer and needs its practice — an entrypoint, a knowledge file, a project skill, a sync or entropy rule, a layer's thickness, the sandbox and threat model, the authorization boundary, the agent's runtime conventions (non-interactive execution, checkpoint commits, worktree exploration), the harness's own self-checks, or more than one agent role — and when asked how such an artifact should be structured or where a rule should live so agents find it; and SHALL not cause it to load for the build workflow itself (investigation, clarification, planning, approval, verification, cleanup), for an application test or CI fix that changes no harness, or for studying another repository's harness.

#### Scenario: Sandbox question
- **WHEN** the user says "how strictly do we need to sandbox the agent that triages incoming issues overnight?"
- **THEN** the skill loads

#### Scenario: Permission list question
- **WHEN** the user says "which commands should our agents be allowed to run without asking, and which never?"
- **THEN** the skill loads

#### Scenario: Checkpoint question
- **WHEN** the user says "write the rule for how agents checkpoint and squash their work on a branch"
- **THEN** the skill loads

#### Scenario: Entrypoint placement question
- **WHEN** the user says "should the release procedure go in AGENTS.md or in a reference?"
- **THEN** the skill loads

#### Scenario: Application CI fix (near-miss)
- **WHEN** the user says "the integration tests time out in CI, fix the workflow"
- **THEN** the skill does not load

#### Scenario: Study of another repository (near-miss)
- **WHEN** the user says "look at how the gateway repo sets up its agents and tell me what is enforced"
- **THEN** the skill does not load

### Requirement: Behavior: Layer calibration uses six layers and five variables
When the build designs or recalibrates several layers, the builder SHALL use six layers — environment and sandbox, knowledge and routing, hard gates, authorization boundary, specification and completion criteria, entropy management — each rated omitted, light, medium, or thick with the light and thick shapes stated, SHALL justify each rating by the five variables (error cost, maintenance horizon, environment isolation, agent autonomy, team size), SHALL start from the exploratory or engineering preset and deviate per layer where a variable differs, and SHALL name the trigger for hardening a graduated prototype and the signal for thinning a constraint that no longer catches anything.

#### Scenario: Preset with per-layer deviation
- **WHEN** the build calibrates a production repository that contains an `experiments/` directory
- **THEN** the plan rates the six layers from the engineering preset and rates hard gates and specification lighter for the experiments area, naming maintenance horizon and error cost as the reasons

#### Scenario: Thinning signal
- **WHEN** the build audits a harness carrying a lint rule written for a bug fixed a year ago that has not fired since
- **THEN** the builder reports the rule as a thinning candidate with the evidence that it no longer catches anything

### Requirement: Behavior: The security boundary is designed from the threat model and audited in the sandbox
When the build reaches the environment layer, the builder SHALL decide isolation by three conditions (sensitive assets or privileges held, externally writable input read, outward action or state change possible; an open-write input source; no human reviewing results before they leave) and require a restricted, credential-stripped instance only when all three hold; SHALL audit the sandbox for the credentials and private files in reach, host privilege leaks (sockets, host root mounts), local and intranet exposure through bridged networking, and where session logs go, and record an explicit accept-or-cut decision per finding; and SHALL have the harness state which control-plane files are hard read-only (security rules, sandbox configuration, secret-scan hooks) and which are soft read-only invariants that may change by proposal but never to make a failing check pass, with any rule made unreadable still rejecting with a readable message that names the policy and the path to change it.

#### Scenario: Sandbox audit
- **WHEN** the build inspects a dev container that mounts the host's `~/.ssh` and the Docker socket
- **THEN** the plan lists both as findings with an accept-or-cut decision each, proposes a scoped token in place of the mounted directory, and records where the framework's session logs are written

#### Scenario: Unreadable rule
- **WHEN** the user wants a command-interception rule whose logic agents must not read
- **THEN** the builder keeps the logic unreadable but writes the rejection message to say it is a policy boundary, not a failure, and to name the alternative or the owner who can change it

### Requirement: Behavior: The authorization boundary is deposited as a reversibility-keyed policy
When the build reaches the authorization layer, the builder SHALL deposit, in the entrypoint's always-applicable part or a knowledge file it points to, a policy that separates capability (sandbox and tokens) from permission; classifies actions by reversibility and outward reach, with irreversible or outward actions pre-authorized in the specification or queued; directs a blocked agent to finish independent work and record pending decisions in a fixed shape (where it is stuck, the options, the recommended option with its reason, the completed work each option affects); writes framework permission lists as explicit allows for known-safe commands and explicit denies for known-dangerous ones with no ask entry lacking a stated reason; treats a refusal as a decision (re-plan, or queue) and never as something to route around; and forbids the agent from relaxing the policy itself.

#### Scenario: Permission list drafted
- **WHEN** the build writes a framework permission file for a project whose sandbox is a dev container
- **THEN** the file allows the project's read-only and planned commands explicitly, denies force push, hard reset, and recursive deletion explicitly, carries no ask entry without a comment giving its reason, and the policy text says the list is fool-proofing against mistakes and the sandbox is the security boundary

#### Scenario: Refusal met during a task
- **WHEN** a deposited policy is read by an agent that has just been refused a write to a protected file
- **THEN** the policy tells it to either re-plan where the logic belongs or queue the item as a pending decision, and forbids retrying with an equivalent command or a script around the check

### Requirement: Behavior: Runtime conventions are deposited for non-interactive execution and commit checkpoints
When the build reaches the runtime conventions of the harness, the builder SHALL deposit non-interactive settings (pager, confirmation prompts, watch modes disabled by environment or flags), a seconds-scale scoped verification command beside the full check, checkpoint rules (a checkpoint is a commit; a task's draft commits are squashed into one atomic commit at task end with a non-interactive recipe; draft commits never leave the machine and pushes name one ref; the secret scan runs on every commit and the pre-commit hook stays seconds-fast; "no draft commits on the branch" is in the task's done criteria; rollback preconditions are enforced by a recipe, not by instruction), worktree exploration with mandatory cleanup, and handoff hygiene (handoff files and session logs live only in untracked locations and are treated as secrets).

#### Scenario: Checkpoint recipe
- **WHEN** the build deposits the checkpoint rule for a project whose agents cannot run interactive rebase
- **THEN** the harness names a non-interactive squash (reset to the task's start commit, then one commit under the project's message convention), states that draft commits are never pushed, and makes the secret scan a per-commit hook

#### Scenario: Handoff file requested
- **WHEN** a deposited harness is read by an agent asked to leave a handoff note for the next session
- **THEN** the harness directs the note to an untracked location and says why (handoff notes carry logs, environment values, and credentials more easily than source)

### Requirement: Behavior: The harness gets deterministic checks of its own
When the project is long-lived, the builder SHALL design harness self-checks in two tiers — reference integrity (entrypoint and knowledge paths exist; documented commands exist as task-runner recipes; mirrored files match) and structural compliance (skill layout and frontmatter, strict specification validation, entrypoint length budget) — run in CI or a hook, written as the project's own management code, never calling a script inside a skill directory, failing with a message that names the file and the fix, and never skipping silently.

#### Scenario: First tier for a small long-lived project
- **WHEN** the build calibrates entropy management as medium for a solo long-lived repository
- **THEN** the plan proposes the reference-integrity tier only, with the task runner as the single command source the check reads, and defers the structural tier until the project has skills or specifications

#### Scenario: Silent skip
- **WHEN** the builder reviews a harness check that skips a file it cannot parse and still exits zero
- **THEN** the builder reports the silent skip as a defect and makes the check fail with the file's name

### Requirement: Behavior: The entrypoint is two-part and splits are justified
The builder SHALL structure the entrypoint as always-applicable constraints first and a situation-keyed routing table after, SHALL before splitting any content behind a pointer answer how the agent will know to read it, what it gains, what unrelated content it ingests, and whether the jump costs less than inlining, SHALL route process-critical skills explicitly from the table, and SHALL keep subdirectory entry files and framework carriers to local supplements that never contradict the root.

#### Scenario: Split proposed without a trigger
- **WHEN** the build considers moving the commit conventions into a knowledge file
- **THEN** the builder states the four answers, finds that the conventions apply to every task, and keeps them in the always-applicable part

#### Scenario: Process-critical skill
- **WHEN** the project has a skill that runs the publish gate before any remote write
- **THEN** the entrypoint's routing table names the situation and the skill, and the design does not rely on the skill's description to fire

### Requirement: Behavior: Multiple agents get isolated resources and one coordinator for remote state
When the plan adopts more than one agent, the builder SHALL isolate concurrent writers by worktree or container, SHALL list the shared resources that need namespacing (ports, test databases or schemas, fixed-path lock files, build caches, external rate limits), SHALL give each agent its own remote branch, SHALL route every write to platform objects (issues, comments, labels, status) through one coordinator with each write pre-authorized, and SHALL name who merges the results and who settles conflicts.

#### Scenario: Two writers, one issue
- **WHEN** the plan has two agents implementing parts of one issue in parallel
- **THEN** it gives each a worktree and a remote branch, routes issue comments through one coordinator, lists the shared test database as a resource to namespace, and names the merger

#### Scenario: Read-only reviewers
- **WHEN** the plan has three review agents reading the same checkout
- **THEN** the builder keeps them in one shared workspace and adds no worktree
