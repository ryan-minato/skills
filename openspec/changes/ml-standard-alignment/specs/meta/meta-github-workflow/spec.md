## ADDED Requirements

### Requirement: Behavior: Research tasks map to pull requests and runs keep the canonical record
When the deposited workflow file records the research profile, the machine-learning reference SHALL implement it: one research task — one objective and one evaluation — maps to one pull request that carries the intent, the source development, the decision, and links to the evidence, while hypotheses and runs stay inside it and metrics stay in the tracker; the reference SHALL state the reachability rule for cited snapshot commits after a squash merge or head-branch deletion (a tag per run or a kept research branch); the committed run record SHALL carry the canonical field list — run id, source snapshot with its retention ref, resolved configuration hash, environment digest, host facts, input identities, randomness, parent run, evaluation identity, metrics with agreed thresholds, artifacts, tracker run, decision; and the tracker default for a new project SHALL be a lightweight open tracker, because the platform provides none.

#### Scenario: Research profile deposited
- **WHEN** the workflow file records the research profile and the builder writes the machine-learning reference
- **THEN** the reference states the one-task-one-pull-request mapping, the snapshot reachability rule, and the tracker default, and the run record template carries every canonical field

#### Scenario: Hypothesis offered as a pull request
- **WHEN** a user asks the deposited guidance whether each hypothesis gets its own pull request
- **THEN** the guidance says a hypothesis is content of a task's pull request, not a request of its own, and points at the hypothesis log
