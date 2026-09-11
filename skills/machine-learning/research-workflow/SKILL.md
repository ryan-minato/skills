---
name: research-workflow
description: >-
  Runs a machine-learning research task end to end — a research spec with
  an objective and an evaluation, a hypothesis loop with snapshot commits
  on an isolated branch, evidence that matches the claim, and a closing
  verdict in one pull or merge request. Use when organizing a series of
  experiments or hypotheses toward an objective: "let's plan the
  experiments", "cut latency by 30% without losing accuracy", "try four
  tokenizer variants"; when writing or revising a research spec; when
  deciding whether results support a claim or one seed is enough; when
  wrapping up a line of research, including with a negative result; or
  when choosing between automatic search and hand-picked runs. Not for
  writing a software specification before implementing a feature, for
  designing a team's work-tracking process, or for recording one run's
  identity.
license: Apache-2.0
---

# Research Workflow

Precedence on every task: an explicit user instruction, then the
project's own conventions and constraints, then this skill's defaults,
then tool preferences. A project that already runs research tasks a
settled way keeps that way.

## The research task

A research task is one **objective** judged by one **evaluation**. It is
the unit of research work and maps to one pull or merge request, which
carries:

- the research intent (the spec, or a link to it);
- the source development the task needed;
- the final decision and its verdict;
- links to the key evidence — the runs, the comparisons, the artifacts.

Inside that request live many hypotheses, many snapshot commits, many
runs, and positive and negative results. Run metrics and run history stay
in the tracker; the request holds decisions, not runs. Do not open one
request per hypothesis: a hypothesis is content of a task, not a task.
When the user asks for a request per hypothesis, explain this once and
follow their reaffirmed decision, recording the deviation.

Humans and agents share the roles fluidly. Typically a human narrows the
solution space — "adjust depth and width to improve efficiency without
losing accuracy" — and the agent proposes hypotheses, changes code or
configuration, runs, analyzes, and continues; sometimes the human's idea
is exact and the agent implements and executes it; sometimes both design
the evaluation together. The spec records the shared understanding, not
a permission boundary.

## The research spec

Write the spec before the first run. Its sections:

| Section | Strength | Holds |
|---|---|---|
| Objective | MUST | what this round improves or verifies, stated as a problem, never as a solution trajectory |
| Evaluation | MUST | the evidence that judges the result; cite the project's existing benchmark rather than restating it |
| Context | SHOULD | what a reader needs to understand the task |
| Search Scope | SHOULD | where answers are currently expected — a hypothesis about the space, not a limit on what may be touched |
| Constraints | OPTIONAL | what must not break: a latency budget, an API, a data rule |
| Completion Condition | OPTIONAL | when enough is known to stop; completion is not success |
| Hypotheses / Notes | OPTIONAL | the current hypotheses and important context |

Any section may be one line. Never fill a section with template prose to
satisfy a shape. Read
[references/research-spec-fields.md](references/research-spec-fields.md)
when writing or revising a research spec. Where the spec lives: inside the
project's specification contract when one exists (a research-task change
under the project's spec tool); otherwise `research/<task>/spec.md` from
[`assets/research-spec.md`](assets/research-spec.md).

The spec evolves; run history does not. When a finding changes the
objective's scope or the search scope, revise the spec and keep earlier
runs attributed to the version they ran under:

```text
spec v1 ── run A, run B
spec v2 ── run C, run D
```

Never pretend A and B ran under v2. When only a solution is offered
("increase depth to 24 layers"), ask for or derive the objective and the
evaluation first, record them, and treat the offered change as the first
hypothesis.

## The loop

```text
research spec
→ hypothesis
→ change code or configuration on the task's branch or worktree
→ snapshot commit
→ run (record identity; log to the tracker)
→ observe and compare against the recorded baseline
→ record the verdict with links to the runs
→ update the spec with what was learned
→ next hypothesis, or close
```

- Work on an isolated experiment branch or worktree; never run research
  on the integration branch.
- Snapshot before every run so the executed source is the recorded
  source; a dirty tree launches nothing.
- Keep a hypothesis log — [`assets/hypothesis-log.md`](assets/hypothesis-log.md)
  is the shape: id, hypothesis, spec version, snapshot commit, runs,
  verdict, evidence.
  It lives with the spec and is the source of the request's summary.
- Read [references/branching-and-pr.md](references/branching-and-pr.md)
  when opening the research worktree or branch, making snapshot commits,
  or preparing the pull or merge request that closes the task.

## Evidence must match the claim

| Claim | Evidence |
|---|---|
| "better" | the quality benchmark the spec names, on its evaluation set |
| "faster" | a performance benchmark on the stated hardware and runtime |
| "more stable" | stability evidence: variance across seeds, spike counts, non-finite events |
| "cheaper" | the resource measurement (tokens, hours, memory) on the stated setup |

For any claim: the evaluation set's identity, the seeds or variance, the
same evaluation code as the baseline, and a recorded baseline run. A
single-seed win by a small margin is inconclusive until repeated or
bounded; downgrade or withhold a claim the evidence does not support.
Read [references/evidence-and-search.md](references/evidence-and-search.md)
when deciding whether results support a claim, when comparing against a
baseline, or when automatic search is on the table.

## Automatic search

When the objective is automatically evaluable and the search space is
explicit, recommend automatic search over hand-tuned trial and error;
humans and the agent define the space, the objective, and the evaluation,
and revise the research question from what the search finds. Call it
hyperparameter optimization only when the space holds hyperparameters
alone; a space that also varies the dataset, the algorithm, the
architecture, or the training strategy searches **search variables**.
Judge the economics: when one run costs days on many accelerators, a
systematic search is not affordable — plan few, deliberate hypotheses and
say so in the spec.

## Closing the task

- Close on the completion condition, not on a positive result. A
  negative or inconclusive verdict backed by evidence is a valid result
  and prevents repetition; record what was ruled out.
- The request's final description states the objective, the verdict,
  the decisive evidence with links, what was ruled out, and what remains
  open. Anything merged to the integration branch is the code the verdict
  needs, squashed or rearranged as the project prefers.
- Every snapshot a recorded run cites stays reachable after the squash:
  apply the project's retention rule (a tag per run, or kept research
  branches) before deleting the branch.
- Archive the spec as the project's contract says; a spec with no
  contract stays in `research/<task>/` with the log.

## Where each fact lives

| Fact | Home |
|---|---|
| Generic research principles | this skill |
| The project's long-lived conventions and navigation | the project's agent guidance and README |
| The current research task | the research spec |
| Run facts, metrics, artifacts | the tracker |
| Executable configuration | the configuration system |
| The software environment | the lock file, the image, the container recipe |

Each fact has one home; every other place points to it.

## Handoffs

- What each run must record and how the tracker is wired is the
  run-identity role. This skill pairs with `experiment-provenance` for it.
  If it is not installed, load the `ryan-minato-skills-installing` skill
  and install `experiment-provenance` as it directs; never run an install
  command yourself. (If that installer skill is absent too, it lives in
  the `core` catalog of https://github.com/ryan-minato/skills.) If the
  user declines, state the minimum inline — the executed commit, the
  resolved configuration, the environment identity, and the input
  identities, with a run id distinct from the commit — and continue.

## Gotchas

- A research spec is not a requirements spec: no scenarios, no acceptance
  criteria beyond the evaluation, no solution steps. Objective and
  evaluation are the whole contract.
- Search Scope is a hypothesis about where the answer lies. Crossing it
  with a reason is research; treating it as an access-control list stalls
  the task.
- A hypothesis is not a work item and not a request; tracking each one in
  the tracker or the board multiplies bookkeeping without adding a
  decision.
- Changing the spec never edits a run: a run belongs to the spec version
  it ran under.
- "Completed" and "succeeded" are different columns: a task closes when
  enough is known.
