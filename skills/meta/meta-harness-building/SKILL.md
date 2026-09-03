---
name: meta-harness-building
description: >-
  Disposable builder skill (delete after the harness is built): the single
  entry point for building, improving, or repairing a project's agent
  harness — learns the repository, settles the requirement with the user,
  plans with approval, builds every layer through the skills at hand, has
  each artifact read back in a clean context, keeps the temporary builders
  out of every commit, and ends by asking whether to remove them before
  review. Use when asked to set up, build, improve, audit, or repair a
  repository's agent setup — AGENTS.md or CLAUDE.md, knowledge files,
  project skills, checks, CI, permissions — or when agents keep missing
  conventions and the harness may have drifted. A project whose topic has
  its own scaffold builder installed starts there; that builder hands off
  here. Not for an isolated
  edit to one file, nor for the practice of each layer, which the
  architecture manual holds.
license: Apache-2.0
compatibility: Live catalog discovery requires Python 3.10+ and network access.
---

# Harness Building

Run every harness build, improvement, or repair through this workflow. The
harness is built once and then read on every future task, so spend context
freely here: load the skills that hold the practice instead of guessing at
it, read the whole repository instead of sampling it, and let independent
readers check the result. A thin build now costs every task later.

This skill holds the procedure only. Practice lives elsewhere: the
methodology in the `meta-harness` skill when it is installed, otherwise in
the `## Harness Methodology` section of `meta-harness-architecture`; the
per-layer practice in `meta-harness-architecture`; and each contract,
platform, container, or language layer in the builder whose description
claims it.

## Rules on every step

- **The builders never enter a commit.** Before the first commit of the
  build, list every disposable builder: each `<skill-root>/<name>/SKILL.md`
  one level below every skill directory the project's frameworks use
  (`.claude/skills`, `.agents/skills`, and any other configured root) whose
  description opens with `Disposable builder skill (delete after the
  harness is built):`. Match that sentence only, never a name or a
  directory. Add each matched directory to
  `$(git rev-parse --git-path info/exclude)`, which stays local and works
  in worktrees; `.gitignore` would ship a rule about skills the project
  will not keep. Stage explicit paths and read `git status` before every
  commit. If a builder was already tracked before this build, say so and
  leave it: its deletion lands with the disposal commit.
- **Nothing in the harness depends on a builder.** No deposited file may
  carry a builder's name, its paths, or the marker sentence above, and no
  term that only a builder defines appears without its definition.
- **A plan is not approval** to publish, change remote settings, delete, or
  grant agent autonomy. Each of those is its own explicit user decision.

## Workflow

### 1. Learn the repository

Read before asking. Record the target and exposed interfaces; layout and
dependency skeleton; technology stack; development, test, deployment, and
CI environments; validation commands; error cost; lifecycle and maintenance
horizon; version-control workflow; maintained sources of truth; and every
existing agent-facing file, skill, registered tool, hook, and check.

For each existing harness artifact, record how a future agent finds it,
when it loads, its source of truth, and what keeps it current; classify
problems as stale, duplicated, contradictory, invisible, missing,
excessive, or orphaned. Load `meta-harness-architecture` and read its
audit reference before judging an existing harness.

Done when: every fact is either backed by inspected evidence or listed as a
question only the team can answer.

### 2. Settle the requirement and the approach with the user

Load the `plan-clarification` skill and interrogate the goal in rounds
until nothing is silently assumed: what the harness must achieve, what is
out of scope, the team facts the repository cannot show (review structure,
confidentiality boundary, how much agents do on their own, who approves
what, which external actions are delegated), what must be preserved, and
the user's preferences. Bring the open questions from step 1 into these
rounds. Do not produce a plan before this converges.

Done when: the user and you state the same goal, scope, and constraints,
and every open question has an answer or a recorded owner.

### 3. Design, exploring the skills at hand

Rate each harness layer with the design axes in
`meta-harness-architecture`: the thickness of every capability and
constraint layer, the evolution mode, agent topology, sync family, model
class, and entropy controls. Record one value and one reason per decision.

Then assign every layer to whatever can build it best. List every skill
available in this session — the installed builders and the durable skills
— and read their descriptions: a layer belongs to the skill whose
description claims it, and that skill's workflow runs the layer and hands
back here when done. This skill names no builder for a layer on purpose;
descriptions route, so a renamed or added builder needs no change here.
When no installed skill claims a layer, run the bundled
[discovery script](scripts/discover.py) to see what the catalog offers:

```bash
python3 scripts/discover.py --catalog meta
```

If a builder for the layer exists but is not installed, load the
`ryan-minato-skills-installing` skill and install the whole `meta` catalog
at project scope as it directs — its builders stack and are disposed
together; never run an install command yourself. If none exists, build the
layer from the manual's references.

Done when: every layer has a thickness, a reason, and an assigned skill or
manual reference.

### 4. Plan and get approval

Present the harness plan grouped by layer, using
[assets/harness-plan-template.md](assets/harness-plan-template.md) only as
a shape; remove every fill instruction and inapplicable row. State the
entrypoint map and every when-to-read route, the feed-forward and feedback
mechanisms, each layer's thickness with its evidence, delegated agent
actions versus human approvals, sync ownership in both directions, the
entropy strategy, the verification criteria, and the assumptions still
open. For an existing harness, plan a gap repair, not a rebuild.

Do not create or change a target artifact before the user approves the
plan.

### 5. Build

Work through the approved plan layer by layer. For each artifact, load the
skill or manual reference assigned in step 3 and follow it; when a builder
finishes its layer it returns here, and this workflow continues. Rework
every copied shape against inspected facts and remove every placeholder.
Commit at meaningful checkpoints under the commit rule above.

Done when: every approved artifact exists, is reachable from the
entrypoint, and contains only real project information.

### 6. Read back in a clean context

Verification is independent reading, not self-assessment. When the
framework can dispatch clean-context subagents, use the smallest model
available for each new or changed artifact: give it only the target
project's files — never a builder's content — and ask it to state, from
those files alone, what the artifact requires of it and when. Compare the
answer with the plan's intent for that artifact; a gap is a defect in the
file, not in the reader. Fix and re-read until the answers match. When
subagents are unavailable, do the same reading yourself in a fresh pass
and record the degradation in the handoff.

Done when: every artifact's independent reading matches its intent.

### 7. Review the finished harness

Review the whole harness once more, in this order:

- **Vocabulary leak.** Terms that only a builder or this catalog defines —
  authority-level names, workflow-contract nouns, skill-authoring jargon,
  entropy-control names, and the like — must not appear in a deposited
  file without their definition. Define each in the harness or rewrite the
  sentence in plain terms; a future agent will not have the builder that
  coined the word.
- **Self-containment.** Have a clean-context agent that has loaded no
  builder (yourself in a fresh pass if none is available) read the
  entrypoint and every file it points to, and confirm it can understand
  and follow the harness with the builders gone. Search the deposited
  files for every builder's name, paths, and the marker sentence; none may
  remain.
- **Mechanics.** Every rule and knowledge source is reachable from the
  entrypoint; the entrypoint stays a map near its budget; local links
  resolve; documented commands run; each sync concern has one owner in
  each direction; the entropy strategy matches the lifecycle.

Done when: all three passes are clean and the fixes they caused were
re-read under step 6.

### 8. Close the build

Report every artifact changed, its role, validation results, and any human
or platform action still required. Then, before the work is presented for
review:

1. Confirm that no builder still has work in this build — none this
   workflow handed off to and the user accepted, none still mid-layer.
2. Check the commit rule once more: `git status` shows no builder
   directory staged or tracked.
3. Ask the user whether to delete the disposable builders now, and wait
   for the decision; the build request was never deletion consent. If the
   user declines or defers, leave the builders in place and excluded,
   record "builders retained" in the handoff, and continue to review.
4. On a decision to delete, load the `meta-disposal` skill and follow it:
   it lists the exact set, confirms the listing, deletes with itself last,
   and clears the exclude entries and dangling links. Commit whatever the
   deletion changed in tracked files.
5. Present the work for review as the project's own workflow directs;
   readying a draft or requesting review is a human decision unless the
   project's authority policy delegates it.

Done when: the harness is verified, the builders are gone or recorded as
retained, no builder is in any commit, and the work sits at the project's
review step.

## Gotchas

- Name categories in durable guidance; enumerate products only where the
  project has actually selected one.
- A document with no discovery path does not exist for future agents.
- A generic "keep docs in sync" rule has no actionable trigger.
- Templates deployed unchanged are unfinished decisions, not scaffolding.
- Saving context is not a goal of this build. The harness is read on every
  future task; an hour of reading now is cheaper than a rule every agent
  misses later.
