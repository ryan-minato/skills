---
name: meta-harness
description: >
  Agent harness methodology — explains, designs, analyzes, audits, improves, and
  repairs everything that makes a project workable for agents. Use for any harness-
  related action: creating or changing AGENTS.md, CLAUDE.md, agent entrypoints,
  development environments, knowledge bases, project skills, tools, permissions,
  approval boundaries, tests, linting, CI, observability, progressive-loading
  structure, synchronization, autonomy, or entropy management; when building a
  complete agent setup or project scaffold; or when agents keep missing conventions
  and the harness may have drifted. For a concrete full-project build, offers
  the disposable meta catalog's harness builders. Not for ordinary application
  test or CI fixes, or public skill authoring, unless the surrounding project
  harness is itself being changed.
license: Apache-2.0
---

# Meta-Harness

Use this methodology for every harness-related action. When the task requires a
complete project investigation, plan, build, or systematic reconstruction, this
skill supplies the invariant method; the build procedure belongs to
`meta-harness-building`, the entry builder of the disposable `meta` catalog,
which routes each layer to that catalog's manual and builders. Offer to
install that catalog as a one-off, project-scope component: load the
`ryan-minato-skills-installing` skill and install the whole `meta` catalog at
project scope as it directs — its builders stack and are removed together once
the harness is verified; never run an install command yourself. If the user
declines, work from this methodology alone.

## Harness Methodology

A harness is everything agent-visible that helps agents meet the project's
expectations: the environment they run in, the information and tools they can
reach, the constraints on what they produce, and the feedback that lets them
correct mistakes. It works in two directions: feed-forward context expands and
guides capability before action; feedback from tests, linting, CI, runtime
observation, review, and users narrows results toward the intended target.
Every boundary that narrows the result also narrows the agent's search for a
rare better one; the balance is set per layer by the five variables below,
never for the whole project at once.

### Research before design

Inspect discoverable project facts before asking questions or choosing artifacts.
Read the goal, structure, stack, commands, checks, deployment environment, existing
instructions, and maintained sources of truth. Ask only for team decisions,
permissions, risk tolerance, and preferences that the project cannot reveal.
Preserve working choices unless the user explicitly requests a migration. Write
specifications only for the parts that are changing: a specification for code
nobody edits rots from the day it is written.

### Write for agents and progressive loading

Agent-first files put facts and direct instructions first, name their source of
truth, state when they load, define checkable completion boundaries, and contain no
unresolved placeholders. Public-convention files such as README, CONTRIBUTING, and
SECURITY remain human-first because their audience is broader. A fact the code
can state — the directory layout, a command, an interface — is not restated in a
document; documents hold what code cannot say, chiefly why and when to read what.

Split content by when it must load, not by abstract topic. Content falls into
five loading modes: needed on every task (the entrypoint's always-applicable
part); needed by task type (a reference or process-critical skill the
entrypoint's routing table names by situation); needed by location (a
subdirectory entry file or path-matched rule); needed by action (a hook that
injects text around a tool call); or recognizable from a description (an
auxiliary skill the framework matches). The last three are framework-dependent:
which file loads when, and which wins, differs between frameworks, so a
subdirectory file or framework carrier holds only local supplements, never a
rule that must hold and never a contradiction of the root. Description matching
is probabilistic, so a skill whose miss has consequences is routed explicitly
from the entrypoint. Before splitting anything behind a pointer, answer four
questions: how the agent will know to read it, what it gains, what unrelated
content it ingests, and whether the jump costs less than inlining. A split that
does not reduce what loads for a task only adds navigation cost. Encode
repeated, fragile, ordered, or branchy work as a project skill; put deterministic
repeated logic in a script that returns a short, actionable result rather than
raw output.

Code that never ships in the product — checks, CI and administration scripts,
hooks, workflow shell, and project-skill scripts — is the project's own
management code. Write it for its role and readable first, and let no fallback
postpone or hide an unexpected error: the failure surfaces with a message that
names what to fix, while a failure the design expects may still be handled or
deferred on purpose. Never bind it to a skill's runtime script: a script inside
a skill's directory, installed or the project's own, is written for the agent
that runs the skill, so give a check, hook, or CI step a script of its own in the
project's paths instead of running the skill's, and treat a rule, check, or
instruction that keeps the two identical as a defect to remove, although matching
content alone is not.

AGENTS.md is the map for progressive loading, not an exhaustive manual. It states
the project's purpose, always-applicable constraints first, validation
entrypoints, and the exact conditions for reading deeper material. Aim for about
100 lines. A light project whose entrypoint is the whole harness may approach 200
lines, but global safety rules stay visible even when that exceeds the budget.

### Place hard boundaries where they are enforced

Defects are kept out of the default branch by CI and git hooks; the environment
and secrets are protected by the sandbox and by scoped tokens; a framework's
permission configuration is fool-proofing against a careless agent, never the
security boundary, because an agent with code execution can route around it.
Prefer a tool's native quiet or compact output flags to custom log parsers, and
accept native noise when no flag exists rather than maintain a fragile parser. A
custom check that rejects something names what to use instead or which document
to read: the rejection is the agent's routing to the fix.

Decide isolation from the threat model, not from fear in general. An agent role
needs a strictly isolated, credential-stripped instance when three conditions
hold at once: it holds sensitive assets or high privileges, reads externally
writable input, and can act outward or change state; the input source is open
to anyone (issues, comments, search results, arbitrary pages) rather than
reputation-guarded or the trusted repository; and no human reviews results
before they leave. Drop any one condition and ordinary protection suffices. A
model's resistance to injection adjusts only the middle tier — reputation-guarded
sources, or open sources with a human in the loop — is confirmed per model, and
never replaces isolation: resistance lowers the chance of hijack, isolation
lowers the damage.

### Separate what agents can do from what they may do

The sandbox and tokens set capability; the authorization boundary sets, inside
that, what proceeds, what waits, and what stays human. Per-command approval is
not security: it fatigues into no review and hands the agent's risk to a human.
Move approval to both ends — the plan approves results and bounds, never a step
list; the end collects decisions the plan did not cover. Reversibility is the
key: reversible internal actions (editing files, branching, local commits,
running tests) proceed; irreversible or outward ones (push, comments, deletion,
publication) are pre-authorized in the specification (or, without one, the
approved plan or the policy file) or queued. A blocked agent
finishes everything that does not depend on the decision and records each
pending item with where it is stuck, the options, the recommended option with
its reason, and the completed work each option affects. A refusal is a decision:
re-plan where the logic belongs, or queue it; never retry with an equivalent
command or a script around the check. Write permission lists as explicit allows
for known-safe commands, explicit denies for known-dangerous ones, and no ask
entry without a stated reason; the rest goes to the framework's default with the
sandbox absorbing a wrong call. The policy forbids the agent from relaxing it.

### Make the runtime non-interactive and checkpointed

Disable pagers, confirmation prompts, and watch modes by environment or flags,
so no agent process hangs on standard input. Provide a seconds-scale scoped
verification command beside the full check. A checkpoint is a commit, never a
stash (unnamed, addressed by stack position, conflict-prone on restore,
invisible in the log) or a framework-only rewind (session-bound, invisible to
humans): a task's draft commits are squashed into one
atomic commit when the task ends, non-interactively; draft commits never leave
the machine and every push names one ref; the secret scan runs on every commit
and the pre-commit hook stays fast enough that nobody bypasses it; "no draft
commits on the branch" is part of a task's done criteria. Explore alternatives
on branches or worktrees and clean them up when the choice is made. Scratch
notes and session logs carry credentials more easily than source: they live only
in untracked locations and are treated as secrets; the pending-decisions list is
not a scratch note and lives where a person will see it.

### Calibrate thickness per layer

Rate six layers separately — environment and sandbox, knowledge and routing,
hard gates, authorization boundary, specification and completion criteria,
entropy management — each omitted, light, medium, or thick. Five variables set
each rating: error cost, maintenance horizon, environment isolation, agent
autonomy, and team size. Start from the preset that fits the work — exploratory
(thin gates, wide allow lists, goals over contracts) or engineering (full gates,
written authorization, scenarios per requirement, self-checks) — and deviate per
layer where a variable differs from the project's overall profile: a research
repository's production pipeline earns engineering gates and entropy management;
a product's experiments directory earns the exploratory column. Never assign one
maturity level to the whole project. Build only what a current need earns, and
name the trigger that would build an omitted layer later.

### Manage harness entropy

Implementation and harness artifacts decay as projects change. Agent-driven
development compresses the change cycle, so contradictions, stale commands,
unreachable knowledge, and excessive constraints can accumulate faster than in a
human-only workflow. A wrong harness is worse than none: without one the agent
reads the code; with a drifted one it builds on a lie. Every harness design,
audit, or improvement must account for the project's lifecycle and choose an
entropy strategy.

There are three complementary controls:

1. Reduce complexity and keep a human in the loop. Leave more judgment to the user
   and agent, reducing the surface that can drift.
2. Add real-time consistency controls. Name concrete forward triggers from
   implementation changes to harness updates and reverse triggers from harness
   edits back to the implementation facts that must be checked; a harness edit
   carries the same verification duty as a code edit. Every fact lives in one
   place and the rest point to it, with a stated rule for which copy wins.
3. Run periodic entropy reclamation. Inspect harness-to-harness and
   harness-to-implementation consistency for stale paths, dead commands,
   duplication, contradictions, unreachable content, and unjustified thickness.
   A constraint that has stopped catching anything is thinned; a prototype that
   graduates to maintained work is hardened in one explicit pass.

A mismatch an agent finds between documents, scripts, and code lands in one of
three places — corrected in the same change when small and certain, queued as a
pending decision when it needs one, or filed as a tracked item with a fixed
label when it is out of scope — and never only in the conversation. A
long-lived harness gets deterministic checks of its own, in two tiers: reference
integrity (paths exist, documented commands exist in the task runner, mirrored
files match) for every long-lived project, and structural compliance (skill
layout, strict specification validation, entrypoint length budget) only once
the project has skills or specifications to check, each failing with the file
and the fix.

Use simplification as the primary control for short-lived, actively supervised
projects. Long-lived or agent-driven projects need real-time consistency controls;
long-lived, high-change projects combine those with periodic reclamation and the
self-checks.

### Make the method durable

Every rule future agents need must land in project-visible files, registered tools,
or reachable sources. Conversation-only decisions do not exist for the next run.
Record team workflow choices without inventing automation, keep one source of truth
per fact, and verify that feed-forward guidance and feedback mechanisms remain
reachable after any temporary harness-building skills are removed.
