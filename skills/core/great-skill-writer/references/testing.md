# Behavioral test-driven skill authoring

Load this only after the invoking-framework gate in SKILL.md passes. Use the
authoring worktree for intended edits and disposable detached git worktrees for
every test run. Never let a test edit the authoring worktree.

## 1. Design the red tests

Before editing, record the base revision and write:

1. At least three realistic prompts that should trigger the description and
   three near-misses that share vocabulary but should not. Include direct and
   indirect phrasing, with the expected load decision for each.
2. Two or three representative outcome tasks and a rubric of observable
   requirements. Mark failures that count as critical regressions.
Keep task text, inputs, model capability, tools, and limits equal across every
solver that will run them.

## 2. Isolate every test run

Create worktrees only immediately before a test run, with outputs outside
version control:

- **Candidate:** detach at the current `HEAD`, then transfer a complete
  temporary snapshot of intended tracked staged and unstaged changes plus
  intended untracked files.
- **Previous version:** for an existing skill, detach at the recorded base
  revision.
- **No skill:** detach at the base revision and make the target skill
  unavailable to that solver.

Give every writing solver its own worktree and output directory. Never let
concurrent writers share a worktree. If isolation, skill exposure, or the
candidate snapshot is incomplete, skip the behavioral tests and report the
specific blocker instead of falling back to the authoring worktree.

## 3. Establish the baselines

Before editing, run the previous-version and no-skill solvers in their test
worktrees. A new skill has no previous version, so omit only that baseline.
The baselines must expose a meaningful gap. If they already satisfy every
requirement, revise the cases or stop: the proposed change has not shown value.

Remove the baseline worktrees after recording their outputs. Recreate fresh
ones for the candidate comparison; do not reuse a solver context.

## 4. Make the tests green

Apply the smallest general skill change that should close the observed gap,
then test the candidate:

- **Trigger accuracy:** use a fresh clean-context subagent for every prompt and
  the invoking framework's load evidence. Give an unexpected result three
  total attempts; it passes only when at least two match the expectation.
- **Outcome quality:** run candidate, previous-version, and no-skill solvers in
  parallel where all three exist. For a new skill, omit only the previous
  version. Retain every output, including failures.
- **Independent grading:** anonymize solver identities and give the outputs,
  rubric, and critical requirements to a clean-context subagent that produced
  none of the answers. Require a score and concrete evidence for every item.
  Accept the candidate only when it has no critical regression and its
  aggregate score is strictly higher than every available baseline.

A tie is not improvement. On failure, fix the underlying instruction rather
than patching one prompt, then rerun the complete affected comparison.

## 5. Clean up and report

Record prompts, rubric, results, scores, evidence, and any skipped test with
its reason. Remove every candidate and baseline test worktree, snapshot,
fixture, harness, and evaluation output. Confirm the authoring worktree contains
only intended changes before continuing.
