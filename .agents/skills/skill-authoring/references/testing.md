# Test-driven skill authoring

Load this only after the invoking-framework gate in SKILL.md passes. It tests
the skill's invocation and effect before static validation; a clean lint result
alone does not prove the skill adds value.

## 1. Create test worktrees only when testing

Record the base revision before editing. The current issue worktree is the
authoring worktree: design, edit, stage, and commit there. Do not create a
test worktree until immediately before a test run.

At the start of each test run, create detached disposable worktrees and
separate output directories:

- **Candidate:** create it at the current `HEAD`, then transfer a complete
  temporary snapshot of the intended current changes. Include tracked staged
  and unstaged changes plus intended untracked files; keep the snapshot patch
  and file copies outside version control.
- **Previous version:** for an existing skill, create it at the recorded base
  revision.
- **No skill:** create it at the recorded base revision with the target skill
  unavailable. A new skill needs only this baseline.

Give every writing subagent its own test worktree and output directory. Never
run tests, generated harnesses, or test-driven edits in the current issue
worktree, and never let concurrent writers share a test worktree. If the
required isolation or complete candidate snapshot cannot be created, skip the
behavioral tests and record the blocker in the Linear milestone comment and
handoff.

## 2. Make the tests red

Before editing the skill, write:

1. At least three realistic prompts that should trigger the description and
   three near-misses that share vocabulary but should not trigger it. Include
   direct and indirect phrasing, with the expected load decision for each.
2. Two or three representative outcome tasks and a rubric of observable
   requirements. Mark failures that count as critical regressions.
3. The previous-version and no-skill baseline outputs when those runs are
   available. Keep task text, inputs, model capability, tools, and limits equal
   across solvers.

The baselines must expose a meaningful gap. If they already satisfy every
requirement, revise the cases or stop: the proposed change has not shown value.

## 3. Make the tests green

Apply the smallest general skill change that should close the observed gap,
then test the candidate:

- **Trigger accuracy:** use a fresh clean-context subagent for each prompt and
  the invoking framework's load evidence. Rerun an unexpected result until it has
  three total attempts; it passes only when at least two attempts match the
  expectation.
- **Outcome quality:** run candidate, previous-version, and no-skill solvers
  in parallel in their test worktrees where all three exist. For a new skill,
  omit only the previous version. Retain every output, including failures.
- **Independent grading:** anonymize solver identities and give the outputs,
  rubric, and critical requirements to a clean-context subagent that produced
  none of the answers. Require a score and concrete evidence for every item.
  Accept the candidate only when it has no critical regression and its
  aggregate score is strictly higher than every available baseline.

A tie is not improvement. On failure, fix the underlying instruction rather
than patching one test prompt, then rerun the complete affected comparison.

## 4. Clean up and report

Record the cases, rubric, results, scores, evidence, and any skipped test with
its reason in the Linear milestone comment and handoff. Remove every detached
candidate and baseline test worktree, snapshot, harness, fixture, and
evaluation output. Keep the current issue worktree for the remaining commit
and PR workflow, and confirm its `git status` shows only the intended repository
changes.
