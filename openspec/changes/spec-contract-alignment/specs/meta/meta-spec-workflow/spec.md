## ADDED Requirements

### Requirement: Behavior: The deposited contract assigns artifact roles, the acceptance shape, and the pre-authorized-actions line
The builder SHALL settle in its questioning round and deposit in the contract an artifact-roles table — for each artifact of a change record, what it owns and what it never holds — with the invariant test (holds under any approach: acceptance's stays-true list; holds only under the chosen approach: design) and the information test (needed again the next time the behavior changes: specification; needed only for this change: design), the acceptance shape (two checkable lists, what becomes true and what stays true, pointing to scenarios for behavior and never restating them), and the line of the record that names the irreversible or outward actions a change pre-authorizes, written in the project's platform vocabulary and pointing to the authority policy for the actions it governs.

#### Scenario: Reading where a fact goes
- **WHEN** a clean-context agent reads only the deposited contract and is asked where to record a rule that must hold whatever approach the implementation takes
- **THEN** it answers the acceptance's stays-true list and names the test that decided it

#### Scenario: Pre-authorized push
- **WHEN** a clean-context agent reads only the deposited contract and the record of a change whose pre-authorized-actions line names one branch
- **THEN** it states that pushing that branch needs no further decision and that any other outward action is queued for the approval owner

### Requirement: Behavior: A spec-as-source surface gets a comparison check
When a project's specification is the source of truth for a machine-readable surface — option metadata, a schema, a command surface, a configuration shape — the builder SHALL record that surface in the contract's source-of-truth table with the check that compares the surface to the specification as part of the project's local check command and its CI, SHALL have the check fail naming the surface and the requirement that differ, and SHALL offer no such check where the specification describes behavior only.

#### Scenario: Option metadata specified
- **WHEN** the project's specifications carry a requirement per option of an artifact's metadata file
- **THEN** the contract names the comparison check, its place in the local check command, and the failure message's contents, and the builder deposits or plans the check as the project's own management code

#### Scenario: Behavior-only specifications
- **WHEN** the project's specifications describe behavior with no machine-readable surface derived from them
- **THEN** the builder records no comparison check and says why

### Requirement: Behavior: Recorded-approval implementations are offered with their costs
When the approval mode question is answered with a recorded approval, the builder SHALL offer, beside the platform's review approval paired with dismissal on new commits, two further implementations with their costs — a label the automation keeps by recomputing the gate's state from the package and withdraws when the package's digest changes or a run crashes, readable locally by one command; and an approval artifact bound to the package digest, the approving actor, and the head commit, with its expiry — SHALL name the property each provides (durable, attributable, bound to one version) and what each costs (code the project owns, re-approval on expiry, forgeability by write-access accounts), and SHALL keep the conversational closing as the default where everyone who needs the decision is in the conversation.

#### Scenario: Audit obligation
- **WHEN** the user says a release manager outside the conversation must verify that the maintainer accepted the frozen version
- **THEN** the builder offers the three implementations with their costs, recommends one from the platform's capability and the project's automation budget, and records the choice per gate

#### Scenario: Solo maintainer in the conversation
- **WHEN** the user is the only reader of the decision and asks for a label-based approval because the reference project has one
- **THEN** the builder recommends the conversational closing, names the code and re-approval cost the label would add, and records a recorded approval only if the user still chooses it

### Requirement: Behavior: Observations are dated and bootstrap narrative is excluded
The builder SHALL write every empirical fact about the platform or the tool in the deposited contract as an observation with its date and scope, labelled as an observation and not a rule, and SHALL keep bootstrap-time narrative — one-time exceptions, readback logs, instructions to execute after a first merge — out of the durable contract, recording such items in the hand-off or tracked work and adding to the contract's "Update this file when" list the trigger that removes a transient exception once its condition is met.

#### Scenario: One-time exception during setup
- **WHEN** the first change of a new project must merge before the protection rule exists and the user asks to record that exception in the contract
- **THEN** the builder records it in the hand-off and tracked work, and the contract carries only the standing rule and a trigger to confirm the protection is active

#### Scenario: Platform behavior seen on a test request
- **WHEN** the builder learns on a test request that the platform's list view does not distinguish a waiting request from a mergeable one
- **THEN** the deposited text states it as an observation with the date and the request it was seen on
