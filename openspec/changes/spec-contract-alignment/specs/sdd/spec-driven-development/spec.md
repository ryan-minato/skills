## ADDED Requirements

### Requirement: Behavior: A specification answers done, non-goals, deliverable, and pre-authorized actions
The agent SHALL have every specification state four things — how completion is judged (acceptance scenarios or a final verification command the implementing agent can execute itself, including the task-level closing conditions the project sets), what the change explicitly does not do, what is delivered (a change request, a report, a deployment, and the pending decisions handed over), and which irreversible or outward actions the change pre-authorizes (which branch is pushed, whether comments or a change request may be opened) — SHALL refuse to treat a specification missing any of them as complete, and SHALL ask for the missing answer in the clarify step rather than assume it.

#### Scenario: Specification without a done criterion
- **WHEN** the user asks the agent to publish a specification whose scenarios are all written but whose completion the agent could only judge by asking the user
- **THEN** the agent names the missing executable criterion, proposes one, and does not publish the package until it is written

#### Scenario: Irreversible action not pre-authorized
- **WHEN** the specification is approved and the implementation reaches a point where the agent would push to a branch the record never named
- **THEN** the agent stops before the push, records it as a pending decision with the options, and continues the work that does not depend on it

#### Scenario: Complete specification
- **WHEN** the user asks whether a specification carrying scenarios, a non-goals list, a deliverable line, and a pre-authorized-actions line is ready for the gate
- **THEN** the agent confirms the four answers are present and moves to the design or the draft

### Requirement: Behavior: Each record artifact owns one role, tested by placement rules
The agent SHALL assign each artifact of a change record one role — the outcome and its acceptance; the behavior contract as requirements and scenarios; the approach's bounds with the information needed only for this change; the steps with their verification — SHALL place every fact by two tests (an invariant that would still hold under any other approach belongs with the acceptance as something that stays true, one that holds only because of the chosen approach belongs in the design; information needed again the next time the behavior changes belongs in the specification, information needed only for this change belongs in the design), SHALL write acceptance as two checkable lists — what becomes true and what stays true — that point to the scenarios for behavior and never restate them, and SHALL name the tool's artifact that carries each role through the framework skill.

#### Scenario: Invariant placed in the design
- **WHEN** the user's draft design states that duplicate emails must always be rejected, a rule that would hold whatever approach is chosen
- **THEN** the agent moves it to the acceptance's stays-true list, leaving in the design only the constraints the chosen approach introduces

#### Scenario: Acceptance restating scenarios
- **WHEN** the user's proposal copies the delta spec's scenarios into its acceptance section
- **THEN** the agent replaces the copies with pointers to the scenarios and keeps in the acceptance only the end state and what stays true

#### Scenario: Information needed again
- **WHEN** the design records the upstream page that defines a protocol the feature implements
- **THEN** the agent moves that source to the specification, because the next change to the feature needs it, and leaves in the design only what this change alone needs
