## ADDED Requirements

### Requirement: Behavior: The research profile names the research task as the unit of tracked work
When the research profile is selected, the builder SHALL name the research task — one objective judged by one evaluation — as the unit of tracked work, SHALL state that the hypotheses and runs inside a task are not work items, SHALL map one task to one change request, and SHALL deposit these in the platform's vocabulary for the platform builder to implement.

#### Scenario: Research profile selected
- **WHEN** the inspection and the first frontier select the research profile
- **THEN** the design summary names the research task as the work unit with hypotheses and runs inside it, and the deposited workflow file says so in the platform's objects

#### Scenario: Hypotheses proposed as work items
- **WHEN** the user proposes tracking every hypothesis as its own work item
- **THEN** the builder states the deletion-test cost once and follows the user's reaffirmed decision, recording it
