# Research Spec Fields

Read when writing or revising a research spec.

## Objective

State the problem: the quantity to improve or the question to verify,
the direction, and the reason it matters now. A solution trajectory
("switch to grouped-query attention") is a hypothesis, not an objective;
move it to Hypotheses and ask what it is meant to achieve.

Good: "Reduce p95 inference latency on the serving hardware by 30% while
keeping the held-out accuracy within 0.5 points of the current release."
Weak: "Try smaller models."

## Evaluation

Name the evidence that judges the result and where it comes from: the
benchmark or evaluation set (by identity), the metric definitions, the
hardware for a performance claim, the seed policy, and the baseline run.
Reference the project's existing benchmark rather than restating how it
is computed. The evaluation is agreed before the first run; changing it
afterwards is a spec revision that earlier runs are not re-judged under.

## Context

What a reader with the project open still needs: the prior result this
task builds on, the constraint that motivated it, the earlier task it
continues, the data or model versions in play.

## Search Scope

Where the answer is currently expected: the components, settings, or
approaches worth varying first, and what is deliberately left alone. It
is a hypothesis about the space. When results point outside it, record
the finding and revise the scope; do not refuse to look.

## Constraints

What must not break: an interface, a latency or memory budget, a data
rule, a compatibility requirement. Constraints are hard; the evaluation
is what is optimized within them.

## Completion Condition

When enough is known to stop: a target met, a bound established, every
hypothesis in scope tested, or a budget exhausted. Completion is not
success — a negative result meeting this condition closes the task.

## Hypotheses / Notes

The current hypotheses, each as one testable statement, and notes that
change how the task is read. The hypothesis log carries their outcomes.

## Approval

Where the project's contract names a gate owner, the gate covers the
objective and the evaluation — not the hypotheses, not the plan. The
spec may evolve after approval; a change to the objective or the
evaluation goes back through the gate.

## Length

Every section may be one line. A section with nothing to say is deleted,
not filled. The spec is read at the start of every session on the task;
its cost is paid every time.
