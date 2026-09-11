## Purpose
Governs what an agent that loaded the `research-workflow` skill observably does when it runs a machine-learning research task: the research spec it writes, the hypothesis loop with snapshot commits, the evidence it requires for a claim, the use of automatic search, and how the task closes.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request concerns organizing a series of experiments or hypotheses toward a research objective, writing or revising a research spec, deciding whether results support a claim, or closing a line of research with a verdict, and SHALL not cause it to load for writing a software specification before implementing a feature, for designing a team's work-tracking process, or for recording one run's identity.

#### Scenario: Organizing a research task
- **WHEN** the user says "we want to cut inference latency by 30% without losing more than 0.5 points of accuracy — let's plan the experiments"
- **THEN** the skill loads

#### Scenario: Closing with a negative result
- **WHEN** the user says "we tried three attention variants and none beat the baseline; how do we wrap this up so nobody repeats it?"
- **THEN** the skill loads

#### Scenario: Software feature spec (near-miss)
- **WHEN** the user says "write the spec for the CSV export feature before we implement it"
- **THEN** the skill does not load

#### Scenario: Team process design (near-miss)
- **WHEN** the user says "design how our team should track issues, sprints, and priorities"
- **THEN** the skill does not load

### Requirement: Behavior: A research spec states the objective and the evaluation before any run
The agent SHALL write a research spec that always carries an Objective (what this round improves or verifies, as a problem, not a solution trajectory) and an Evaluation (the evidence that judges the result, referencing an existing benchmark when one exists), SHALL add Context and Search Scope when known, Constraints, Completion Condition, and Hypotheses or Notes when useful, SHALL keep every section as short as the information allows with no template filler, and SHALL treat Search Scope as a hypothesis about where answers lie rather than as a permission boundary.

#### Scenario: Spec from a stated objective
- **WHEN** the user gives an objective and an evaluation metric and asks the agent to start
- **THEN** the agent writes a spec with Objective and Evaluation, adds Search Scope from the conversation, omits empty sections, and does not run anything before the spec exists

#### Scenario: Only a solution is offered
- **WHEN** the user says "increase depth to 24 layers and see what happens" with no objective stated
- **THEN** the agent asks for or derives the objective and the evaluation, records them, and only then treats the depth change as the first hypothesis

#### Scenario: Search scope proves wrong
- **WHEN** results show the answer lies outside the recorded Search Scope
- **THEN** the agent updates the spec's Search Scope with the finding rather than treating the scope as a limit it may not cross

### Requirement: Behavior: One research task maps to one pull or merge request that carries decisions, not runs
The agent SHALL run one research task — one objective and one evaluation — inside one pull or merge request that carries the research intent, the source development, the final decision, and links to the key evidence, SHALL keep hypotheses and runs inside that request rather than opening one per hypothesis, SHALL leave run metrics and history to the tracker, and SHALL explain the default once when the user asks for a request per hypothesis and follow the user's reaffirmed decision while recording it.

#### Scenario: Several hypotheses under one objective
- **WHEN** the task has four hypotheses to test
- **THEN** the agent keeps all four on the task's branch and request, records each hypothesis with its snapshot commit, runs, and verdict, and does not open four requests

#### Scenario: Request per hypothesis requested
- **WHEN** the user asks for a separate pull request for each hypothesis
- **THEN** the agent explains once that a request carries a decision about one objective and that hypotheses are its contents, and follows the user's decision if they reaffirm it, recording the deviation

### Requirement: Behavior: The hypothesis loop snapshots before every run and never rewrites run history
For each hypothesis the agent SHALL work on an isolated experiment branch or worktree, make a snapshot commit before each run so the executed source is the recorded source, compare results against a recorded baseline, record the verdict with links to the runs, and update the spec with new findings, and SHALL leave earlier runs attributed to the spec version under which they ran when the spec changes.

#### Scenario: Run launched after edits
- **WHEN** the agent has changed the model code for a hypothesis and is about to launch
- **THEN** it commits a snapshot on the experiment branch, launches, and records the run against that commit

#### Scenario: Spec revised mid-task
- **WHEN** a finding changes the Objective's scope after two runs completed
- **THEN** the agent revises the spec and records that the two earlier runs answered the earlier version, without editing their records

### Requirement: Behavior: Evidence matches the claim and completion is not success
The agent SHALL require evidence of the same kind as the claim — a quality benchmark for "better", a performance benchmark on the stated hardware for "faster", stability evidence for "more stable" — with the evaluation set identity, the seeds or variance, and the same evaluation code as the baseline, SHALL downgrade or withhold a claim the evidence does not support, and SHALL close a task on its completion condition, recording a negative or inconclusive verdict as a valid result.

#### Scenario: Improvement claimed from one seed
- **WHEN** the user wants to declare a hypothesis a win from a single-seed run that beats the baseline by a small margin
- **THEN** the agent asks for repeated seeds or variance evidence before the claim, or records the result as inconclusive

#### Scenario: Negative result
- **WHEN** every hypothesis failed the evaluation and the completion condition is met
- **THEN** the agent closes the task with a recorded negative verdict, the evidence links, and what was ruled out, and does not keep the task open waiting for a positive result

### Requirement: Behavior: Automatic search is recommended when the space is defined and the compute allows it
When the objective is automatically evaluable and the search space is explicit, the agent SHALL recommend automatic search over hand-tuned trial and error, SHALL call the search hyperparameter optimization only when the space contains hyperparameters alone and otherwise name the searched values search variables, SHALL keep the definition of the space, the objective, and the evaluation with the humans and the agent, and SHALL judge the economics: when one run is so expensive that a systematic search cannot be afforded, it SHALL plan few, deliberate runs instead.

#### Scenario: Cheap runs, explicit space
- **WHEN** each run takes minutes and the user lists five values to try for two settings
- **THEN** the agent recommends an automatic search over that space with the recorded evaluation as the objective

#### Scenario: Large-model scale
- **WHEN** each run takes days on many accelerators
- **THEN** the agent recommends a small number of hypotheses chosen by reasoning, states why a sweep is not economical, and records that choice in the spec

### Requirement: Handoff: run identity and recording
When the user asks what each run must record or how the tracker should be wired, the agent SHALL offer the run-identity role through the installing skill without printing an install command, and SHALL, when the user declines, state the minimum inline — the executed commit, the resolved configuration, the environment identity, and the input identities, with a run id distinct from the commit — and continue the task.

#### Scenario: Handoff offered
- **WHEN** the user asks, during a research task, what the run record must contain
- **THEN** the agent names the run-identity role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent states the four-part minimum with the run id and continues the research task
