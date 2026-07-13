# Test-driven skill authoring

Use this workflow for every skill created or modified in this repository. It
tests the skill's invocation and effect before static validation; a clean lint
result alone does not prove the skill adds value.

## 1. Gate on capabilities

Discover capabilities instead of inferring them from an agent environment's
name. Trigger and outcome tests require all of these:

- subagents that start with context isolated from the authoring session;
- a way to make the candidate skill available to one run and unavailable to
  another while keeping other conditions equal;
- for trigger tests, observable evidence that a run loaded the skill body.

The evidence format belongs to the environment. Do not prescribe a particular
agent API, configuration path, event name, or log shape. If a capability is
missing, record the skipped test and reason in the Linear milestone comment and
handoff. Never use the authoring agent as a substitute for an independent
subagent. Script tests in step 5 do not require subagents.

## 2. Isolate the baselines

The `issue-workflow` skill creates the candidate issue branch in its own
worktree before this workflow begins. Record its base revision. For an existing
skill, add a detached worktree at that revision for the previous-version
baseline. Add another detached worktree at the same revision for the no-skill
baseline, with the target skill unavailable to that run. A new skill needs only
the no-skill baseline.

Give every writing subagent a separate worktree and output directory. Never run
tests or skill edits in the primary working tree, and never let concurrent
writers share a worktree. If the required isolation cannot be created, stop and
report the blocker.

## 3. Make the tests red

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

## 4. Make the tests green

Apply the smallest general skill change that should close the observed gap,
then test the candidate:

- **Trigger accuracy:** use a fresh clean-context subagent for each prompt and
  the environment's load evidence. Rerun an unexpected result until it has
  three total attempts; it passes only when at least two attempts match the
  expectation.
- **Outcome quality:** run candidate, previous-version, and no-skill solvers in
  parallel where all three exist. For a new skill, omit only the previous
  version. Retain every output, including failures.
- **Independent grading:** anonymize solver identities and give the outputs,
  rubric, and critical requirements to a clean-context subagent that produced
  none of the answers. Require a score and concrete evidence for every item.
  Accept the candidate only when it has no critical regression and its
  aggregate score is strictly higher than every available baseline.

A tie is not improvement. On failure, fix the underlying instruction rather
than patching one test prompt, then rerun the complete affected comparison.

## 5. Test bundled scripts without subagents

For every added or changed bundled script, generate an untracked temporary test
harness inside the candidate worktree and run it with the declared runtime. It
must verify:

- `--help` exits 0 and shows a usage example;
- a representative invocation exits 0 with the expected output;
- repeating the invocation proves idempotence;
- bad arguments exit 2 with an actionable diagnostic.

Add domain-specific cases when they protect correctness. Record the commands
and results in the Linear milestone comment and handoff, then delete the harness,
fixtures, and evaluation outputs before staging.

## 6. Clean up

Keep the candidate issue worktree for the remaining commit and PR workflow.
After recording results, remove the detached baseline worktrees and confirm
that `git status` shows only the intended repository changes.
