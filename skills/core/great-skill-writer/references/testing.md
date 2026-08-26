# Behavioral test-driven skill authoring

Load this only when SKILL.md confirms clean-context subagents are available.
Use a disposable detached candidate worktree for each test run when possible.
Never let a test edit the authoring worktree.

## 1. Design acceptance tests before editing

Before editing, write:

1. At least three realistic prompts that should trigger the description and
   three near-misses that share vocabulary but should not. Include direct and
   indirect phrasing, with the expected load decision for each.
2. Two or three representative outcome tasks and a rubric of observable
   requirements. Mark critical failures and declare the aggregate score needed
   to pass at that tier before seeing candidate output.

Run every solver — trigger accuracy and outcome quality alike — on the least
capable model tier the skill is meant to support (the weakest tier the
harness offers at or above that floor), and keep task text, inputs, tools,
and limits equal across all of them: a stronger solver clears the rubric on
its own priors rather than on the skill, certifying one that breaks below the
tier it claims — and the floor tier is the cheaper run. When the harness
fixes the solver model or a run escalates the tier, acceptance certifies only
the tier that ran.

Use framework-native conversation history or skill-load telemetry to determine
whether a solver loaded the target. If neither is available, append this neutral
instrumentation to every solver prompt without naming the target skill:

```text
End your answer with SKILLS_LOADED: <comma-separated skill names or none>.
List only skills whose bodies you actually loaded; do not load a skill merely
to report it.
```

## 2. Isolate every test run when possible

Create the candidate worktree only immediately before a test run, with
outputs outside version control: detach at the current `HEAD`, then transfer
a complete temporary snapshot of intended tracked staged and unstaged
changes plus intended untracked files.

Make the candidate visible to each solver's normal skill discovery. When it
cannot be exposed, record the degradation and note which version of the
target — installed or candidate — the observation actually reflects; the
results apply only to that version.

Give every writing solver its own worktree and output directory. Never let
concurrent writers share a worktree. If a worktree or complete candidate
snapshot is unavailable, use the best available environment, keep generated
material outside version control, and record the isolation degradation.
Never explicitly tell a trigger-test solver to use the target skill.

## 3. Evaluate the candidate

With the intended change in place, test the candidate:

- **Trigger accuracy:** use a fresh clean-context subagent for every attempt.
  Determine loading with the observation mechanism selected in section 1; in
  the fallback report, the target name present in `SKILLS_LOADED` means
  loaded, absent means not loaded. A missing observation or malformed
  fallback report invalidates the attempt. Run each case at least twice, up
  to three attempts total: the case passes when two valid attempts match the
  expected load decision, fails when two valid attempts contradict it, and
  is otherwise skipped and reported as inadequate observability.
- **Outcome quality:** run a candidate solver for every representative task in
  its own test worktree. Retain every output, including failures. An output is
  valid only when the selected observation mechanism shows the target loaded.
  Apply the same three-attempt limit, then skip and report if a valid output
  cannot be obtained.
- **Independent grading:** anonymize solver identities and give the outputs,
  rubric, and critical requirements to a clean-context subagent that produced
  none of the answers. Grade on a model capable enough to apply the rubric
  reliably: the floor rule binds solvers, not judges — a weak grader turns
  every score into noise. Require a score and concrete evidence for every
  item. Accept the candidate only when it has no critical failure and its
  aggregate score meets the threshold declared before the run.

On failure, fix the underlying instruction rather than patching one prompt,
then rerun the complete affected evaluation. When a floor-tier solver cannot
complete the task at all with the target loaded, the finding is the claimed
floor, not the attempt: strengthen the instruction to carry that tier, or
escalate one tier and rerun. Scoring badly is an instruction failure, never a
reason to escalate.

## 4. Clean up and report

Record prompts, observation method, solver tier, rubric, results, scores,
evidence, every isolation degradation, and any skipped test with its reason.
Remove every candidate test worktree, snapshot, fixture, harness, and
evaluation output. Confirm the authoring worktree contains only intended
changes before continuing.
