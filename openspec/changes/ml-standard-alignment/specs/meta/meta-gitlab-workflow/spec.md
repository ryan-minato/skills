## ADDED Requirements

### Requirement: Behavior: Research tasks map to merge requests and runs keep the canonical record
When the deposited workflow file records the research profile, the MLOps reference SHALL implement it: one research task — one objective and one evaluation — maps to one merge request that carries the intent, the source development, the decision, and links to the evidence, while hypotheses and runs stay inside it and metrics stay in the tracker; the reference SHALL state the reachability rule for cited snapshot commits when the merge squashes or deletes the source branch; the experiment identity list SHALL carry the canonical field list — run id, source snapshot with its retention ref, resolved configuration hash, environment digest, host facts, input identities, randomness, parent run, evaluation identity, metrics with agreed thresholds, artifacts, tracker run, decision; and the tracker default for a new project SHALL be the platform's experiment tracking when the instance provides it, else a lightweight open tracker.

#### Scenario: Research profile deposited
- **WHEN** the workflow file records the research profile and the builder writes the MLOps reference
- **THEN** the reference states the one-task-one-merge-request mapping, the source-branch reachability rule, and the tracker precedence, and the identity list carries every canonical field

#### Scenario: Instance without experiment tracking
- **WHEN** the instance's experiment-tracking feature is unavailable
- **THEN** the guidance falls back to a lightweight open tracker and records the check that found the feature unavailable
