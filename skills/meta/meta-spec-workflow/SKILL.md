---
name: meta-spec-workflow
description: >-
  Disposable builder skill (delete after the harness is built): configures
  a project's spec-driven development rules — settles level, tool, request
  shape, approval package and gate modes, the freeze, and scope with the
  user, gives every fact one source of truth, deposits the specification
  contract in the project's platform vocabulary, and hands tool adoption
  and automation to the tool's framework skill; in a second phase, once the
  platform base is delivered, fills its extension slots (templates, forms,
  project skill steps, knowledge). Use in a harness build when the workflow
  contract records spec-driven intent, when a spec tool's directories sit
  beside an agent harness, when asked to make the harness match the spec
  tool or stop issues and specs contradicting each other, or when a
  delivered platform base awaits its specification shaping. Not for
  writing specs, choosing the practice, the loop, the tracker, branching,
  or one framework's automation.
license: Apache-2.0
compatibility: >-
  The bundled script requires Python 3.10+ (stdlib only).
---

# Specification Workflow Contract

Settle with the human developers how this project writes, approves, and
keeps its specifications, make the harness say the same thing the chosen
tool assumes, and leave the result in the target project as a contract.
Then, once the platform builder has delivered its paradigm-neutral base,
come back and shape that base for the contract: the templates, the intake
forms, the project skill's steps, the knowledge section — and hand the
tool's automation and validator wiring to the tool's framework skill. The
design is settled in tool and workflow terms with no hosting platform in
mind; the deposited contract and the inserted slot text name the
platform's objects, because the agents that read them work on that
platform. No framework command, script, or workflow appears in either.
Both must survive this builder and this conversation.

## Non-negotiable boundaries

- The contract records specification discipline only: level, tool, artifact
  map, source of truth per fact, lifecycle, approval gate, and the division
  of labor with tracked work. It adds no management semantic — a
  specification is the content of a change request's acceptance, never a
  new work-item kind, objective, or timebox — and it never reopens a
  decision the workflow contract already settled.
- The design is tool-neutral and platform-neutral; the deposit is not.
  Questions and the design summary speak of specifications, change
  records, tracked work, and change requests so that no platform habit
  shapes the discipline; the deposited contract, after approval, names the
  platform's objects and operations, because it is read on that platform
  every day.
- Tool-owned files are public-convention files. A constitution, a spec
  directory, a steering file, or a change record follows the tool's format
  and conventions; never rewrite one agent-first, never add harness
  vocabulary to it, and never "fix" a tool's lowercase `design.md` into the
  reserved `DESIGN.md`.
- One source of truth per fact. Where a tool-owned file and the entrypoint
  or knowledge base state the same fact — engineering principles, behavior
  of a domain, acceptance of a change — exactly one keeps it and the other
  points to it. Restating is the failure this builder exists to prevent.
- Framework content lives in the framework skill. This builder names no
  tool command, layout detail, script, or workflow; the `sdd` catalog's
  framework skill for the selected approach owns them, and tool commands
  are verified from the tool's own `--help` there, never recalled.
- A recommendation is not a decision. Attach one reasoned recommendation
  to every question and let the user overrule it; a settled decision becomes
  a downstream constraint later builders must not reopen.
- Nothing lands before agreement. Keep the working design in the
  conversation; write into the target project only after the user approves
  the complete design summary. Preserve working conventions — a spec tool
  the project already runs is kept, not migrated.
- Disposable builders never enter a commit. Before the first commit of the
  build, add every skill directory whose description opens with
  `Disposable builder skill (delete after the harness is built):` to
  `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and read
  `git status` before each commit; a builder tracked before the build is
  reported, and its deletion lands with the disposal commit.

## Workflow

### 1. Inspect the project

Run [`scripts/detect_spec_tooling.py`](scripts/detect_spec_tooling.py) in
the target checkout for a read-only evidence sweep:

```bash
python3 scripts/detect_spec_tooling.py --root .
```

It reports which spec tool layouts exist, how many specs and change records
each holds, the tool-owned paths, every agent-entrypoint line that points at
a spec artifact, and every document that looks like it restates
requirements or acceptance criteria; installed agent skills are skipped,
because their instructions are not project requirements. Quote its summary line in the
inspection record. The script recognizes requirement-shaped text only; a
plain sentence in the entrypoint that states what the system does ("five
failed attempts lock the account") is invisible to it, so read the
entrypoint and every knowledge file in full against the tool's
source-of-truth specs and list each behavior statement they hold.

Then read the upstream contracts, when present, at the paths the
entrypoint's pointers record — the workflow file
(`.agents/knowledge/<platform>-workflow.md`, for example
`github-workflow.md`: its pull or merge request and acceptance rules and
its recorded change propagation are the anchors this contract attaches to,
and its platform is the vocabulary this contract is written in),
`.agents/knowledge/git-workflow.md` (feature directories and branch names
must follow it, never a tool default), and the project's goal document.
When no workflow file evidences the platform, establish it from the remote
and the CI or template directories; if nothing does, ask in step 2 before
anything else, because no contract is deposited without it. Record also
whether automation exists (a workflow or pipeline directory): that
evidence decides which read-only automation is worth installing. Record
whether a platform base is already
delivered — a project workflow skill, request and intake templates, a
mechanics section in the workflow file — and whether this contract is
already deposited: with both present, the run is the second phase and
starts at step 7 after this inspection. Sort everything into known facts
with evidence, unknown facts still discoverable, and decisions only a
human can make.

Done when: the script's summary is recorded; every existing spec artifact,
every requirement-bearing document, every entrypoint pointer, and every
behavior statement in the entrypoint or knowledge base is listed with its
source of truth; and the human-decision list contains nothing an
inspection could resolve.

### 2. Settle the level and the approach

Ask the whole set in one numbered round, each question with one reasoned
recommendation:

1. **Level** — spec-first (spec used to build, then archived),
   spec-anchored (spec kept as the living description; default for anything
   maintained past one release), or spec-as-source (only when the user names
   it and the toolchain regenerates code).
2. **Approach** — one default by situation, the rest as deviations: no code
   yet and an application delivered feature by feature → the spec-first kit
   (GitHub Spec-Kit); no code yet and a library, framework, or
   infrastructure → the spec-anchored change workflow (OpenSpec), whose
   per-capability specs match a library's contract better than
   feature-shaped deliveries; existing code, whatever its shape → the
   spec-anchored change workflow (OpenSpec); the team works inside Kiro →
   its native spec format, with the unverified portability of those specs
   to other agents stated; the team refuses tooling → committed
   specification documents; custom layout only for a stated constraint none
   of these meets. A tool the project already runs is the answer unless the
   user asks to change it. Record which framework skill of the `sdd`
   catalog owns the tool (`openspec-workflow`, `spec-kit-workflow`), or
   that none exists for it.
3. **Approval gates, mode, and package** — who approves, whether an agent
   may approve its own, how each approval is recorded, and when a design
   is warranted. There are two gates: the first on the complete approval
   package before the task list, the second on the finished
   implementation before the record is frozen. The package is the
   specification plus the design when one is warranted — by default when
   more than one reasonable approach exists, or the change touches
   structure, interfaces, dependencies, or files outside the record; a
   wording change inside one section needs none; the project may fix the
   rule in its schema. The design bounds the approach and lists no steps;
   it is committed, so it carries no secret or private data. Recommend
   closing both gates in conversation: the gate owner discusses on the
   request, directs changes in conversation, and declares each
   deliberation closed; the agent reconciles the request's threads before
   acting. Offer a **recorded approval** — a durable, attributable mark
   naming one version — per gate, and derive the recommendation from who
   reads the record: propose one only when someone outside the
   conversation must verify for themselves that a named person accepted a
   named version, never from team size or pipeline maturity. Both modes
   are defined in
   [contract-design.md](references/contract-design.md), read before this
   round. Record the owner, the mode of each gate, and the design rule;
   the authority builder attaches its levels to these gates later.
4. **Division of labor** — confirm the default: specifications own what,
   why, and acceptance; tracked work owns who, when, and status and links
   the specification. Any deviation is recorded with its reason.
5. **Change request shape** — combined (one change request carries the
   change record from the moment it is committed, opened as a draft, and
   the approval gate is exercised on that draft before implementation) or
   split (a specification change request carries only the record, is
   approved and merged, and implementation change requests follow). Derive
   the recommendation from the change propagation the workflow file
   records and cite that line: local, copied, or state-mutating propagation
   → combined; dependency or inherited propagation, a standards project, or
   any consumer that depends on a stable contract → split. Any project may
   take a single contract-level change through split as a recorded
   deviation. Record the answer where the framework skill's check can read
   it: the shape decides whether any request may merge with an unfrozen
   record, and under split exactly one may — the specification request,
   which implements nothing. Under combined none may, so a request that
   implemented nothing and still holds an unfrozen record is unfinished,
   not exempt.
6. **Archive executor and the freeze** — archiving (or converging)
   always happens inside the change request, and it is the freeze the
   second gate's approval applies to: the request is marked ready with
   the record still open, the deliberation runs against that, and the
   freeze follows it. The executor is a person who holds the branch —
   the implementer, or a maintainer who pulled a fork's branch. No job
   archives: no platform token can push to a fork, so such a job could
   never serve an external contribution, and on a branch it could reach
   it would be the only automation needing write access to the
   repository's contents. What automation the framework skill does
   install is read-only and exists for visibility: a required check that
   fails a ready request holding an unarchived record — the red that
   blocks the merge through the whole deliberation — comment commands
   that print a record into the discussion thread, and status labels that
   make the state legible from the list view. Derive from step 1's
   evidence which of those the project takes, and record how a spent
   freeze is detected: a commit after the freeze commit means the
   approved version no longer exists. Where the tool has no archive
   operation, the freeze is a declaration on the request naming the
   commit the record stands at. Archiving after the merge by a job that
   pushes to the integration branch is a rejected alternative, not an
   option.
7. **Default specification author** — the implementer, or a named planning
   role. Record who publishes the draft and who approves it (the approval
   owner of question 3); the record of approval is the mode's closing or
   comment, never the platform's review-approval state, which later pushes
   dismiss.
8. **Specification scope** — recommend that domains cover the product the
   project delivers and that a change to the project's own harness,
   tooling, checks, workflows, or documents is a spec-less change carried
   by the tool's marker (the framework skill names it; the change carries
   a proposal, a design, and tasks and no delta spec), linked by tracked
   work the same way and never given a domain. Record any deviation with
   its reason.

Done when: level, approach, approval owner and mode, division of labor,
change request shape, archive executor, default author, and specification
scope are settled by the user or confirmed from evidence, each with its
selecting fact recorded.

### 3. Install or adopt the tool

Hand the installation or adoption to the framework skill for the selected
approach — the `sdd` catalog's `openspec-workflow` or `spec-kit-workflow`
of https://github.com/ryan-minato/skills. If it is not installed, load the
`ryan-minato-skills-installing` skill and install it as it directs; never
run an install command yourself. When the user declines, or no framework
skill exists for the approach (an IDE's native spec files, committed
documents, a custom layout), apply only the generic rule below and record
the by-hand executor.

Before any tool initializer runs in an existing repository, record the
current tree (`git status --porcelain` and a file listing of every agent
directory the tool may write into — `.claude/`, `.github/`, `.agents/`,
`.kiro/`, and the tool's own directory). Run the initializer only with the
user's approval, then diff: every file it created or changed is kept,
merged into an existing harness file, or removed — with the decision
recorded. An initializer that rewrote an entrypoint or dropped a second
command set beside an existing one has created a duplicate, not a setup.

Done when: the tool's layout exists at the agreed paths, every file the
initializer touched has a recorded disposition, and no harness file was
silently replaced.

### 4. Reconcile the harness

- The entrypoint gains one event-triggered pointer to the contract (see
  step 5) and, where the tool needs it, one pointer to the tool-owned
  source of truth. It restates nothing from either: every behavior
  statement the inspection listed in the entrypoint is deleted or replaced
  by that pointer, with the user's approval — the entrypoint is the most
  common place for a spec-owned fact to survive in duplicate.
- Every knowledge-base passage that describes system behavior a spec now
  owns becomes a pointer to that spec; every engineering principle the
  tool's constitution or steering now owns moves there, and the harness
  points to it.
- Feature directory and branch naming follow the branching contract; where
  the tool numbers or names directories, record the mapping between a spec
  directory and its branch instead of letting two schemes coexist.
- Every requirement-bearing document the inspection flagged — a project
  document, never an installed skill's instruction file — becomes a spec, a
  pointer to a spec, or a deletion, each with the user's approval.
- The goal document stays above the constitution or steering: they may cite
  it, never restate it.

Done when: every behavior statement and requirement-bearing document the
inspection listed is now a pointer, a spec, or a deletion — or carries a
recorded, user-approved reason to stay — and the entrypoint states no
behavior a spec owns.

### 5. Deposit the contract

Read [durable-output.md](references/durable-output.md) on every build.
Adapt [assets/spec-workflow.md](assets/spec-workflow.md) to the settled
answers: it is a raw shape, and every placeholder and inapplicable section
must be gone. Wire the entrypoint pointer with the events that trigger
reading the contract.

Done when: the contract lives in the target project; it names the level,
tool, artifact map, source-of-truth table, lifecycle, approval gate,
division of labor, and update triggers, each in the platform's own
vocabulary (issue or work item, pull or merge request, draft, the workflow
or job that archives); and this model-vocabulary check over the deposited
file returns nothing:

    grep -inE 'tracked work|work item kind|change request|draft change|integration branch' <file>

Tool names (the spec kit, the change workflow) are facts and appear as
they are; a term only this builder defines is a design word that leaked
and is replaced by the platform's object, never waived.

### 6. Verify and hand off

Simulate removal: with this builder deleted, the next agent must be able to
name the level, the tool, where a new specification goes, which file rules
on a behavior, who approves a spec and where that approval is recorded,
when the draft opens, whether a spec needs its own change request, who
archives and when, and how a work item links a spec, from target-project
files alone. Confirm the deposited file does not carry this
skill's disposable marker, name, or paths.

Then hand off by name, in order. Governance next: attach the approval gate
and authority levels with `meta-agent-authority`. Platform base after
that: `meta-github-workflow` or `meta-gitlab-workflow` for the evidenced
host delivers a paradigm-neutral base with extension slots and knows
nothing of this contract beyond its existence. Then return here for step
7, which shapes that base. If either builder is not installed, load the
`ryan-minato-skills-installing` skill and install the whole `meta` catalog
at project scope as it directs — its builders stack and are disposed
together; never run an install command yourself.

If the user declines, record in the hand-off which decisions remain
unexpressed on the platform and end the run after the contract deposit,
naming the platform builder and this builder's step 7 as the remaining
steps. Every hand-off report ends with the order above and with the
closing step that follows.

### 7. Shape the platform base

Run this step only when the base exists: the project workflow skill, the
request and intake templates, and the mechanics section of the workflow
file are in place. Read the expression reference for the evidenced
platform — [github-expression.md](references/github-expression.md) or
[gitlab-expression.md](references/gitlab-expression.md) — and the base's
own slot register (its durable-harness reference, `## Extension slots`).
Fill the slots in this order, each from the matching section of
`assets/<platform>/`, under the fill contract the reference states
(locate by structure, insert only, grep first so a second run changes
nothing, never touch the security or sensitivity-review checklist item):

1. Templates and forms: the six template slots, worded for the contract's
   approval package, the mode of each gate, and the freeze.
2. The project skill: the four step slots, worded for the shape and the
   modes, including the reconciliation each gate requires.
3. The request automation and the validator: hand both to the framework
   skill (the handoff of step 3), which installs the check into the
   command and workflow the base already runs, the comment commands, the
   status labels, and its script, and records
   the maintainer actions those need; edit no workflow or pipeline file
   yourself. Without a framework skill, record the by-hand executor, the
   request checklist as the gate, and the automation as remaining work.
4. Knowledge: `KNOWLEDGE_SECTION` in the platform workflow file, one
   `SYNC_ROW` per insertion, and the `MAINTAINER_ACTION` rows the
   framework skill names (the label sync, and any token variable its
   read-only jobs need).

Done when: every slot the contract's shape and modes require is filled;
`grep -rn '{{[A-Z]'` over the delivered paths returns nothing (the
uppercase form is the builders' placeholder; Actions expressions stay); a body built
from the shaped template passes the base's checklist check; the framework
skill has been handed the automation or the by-hand executor is recorded;
and a clean-context read of the project skill can state the precondition,
first content, reconciliation, and finish step.

### 8. Close

When this builder runs under `meta-harness-building`, return there for the
closing step. When it runs alone, once the deposit — and, in the second
phase, the shaped base — is verified and before the work goes to review,
ask the user whether to delete the disposable builders now — the build
request is not deletion consent — and on that decision load
`meta-disposal`, which lists, confirms, and removes them. If the user
declines, leave the builders in place and out of every commit, and record
it in the handoff.

## Gotchas

- Spec tools rename commands and move files between releases; the
  contract records paths the build verified, with the date, never a command
  list.
- The spec-first kit's feature script creates a numbered spec directory and
  no git branch; a harness that says "the tool creates the branch" sends
  every agent onto the wrong ref.
- A constitution or steering file is read by the tool at plan time. Copying
  it into the entrypoint makes the entrypoint the copy that drifts; the
  entrypoint points, the tool-owned file rules.
- A change workflow's main specs describe behavior; the knowledge base
  describes conventions and mechanics. A knowledge file that narrates what
  the system does is a spec in the wrong place.
- Backfilling specs for code no change touches feels like progress and
  produces specs nothing keeps honest; the contract says specs cover changed
  behavior only, and the codebase map covers the rest.
- Acceptance criteria copied into an intake template are the single most
  common contradiction source; the second phase makes the template link
  the spec's scenarios instead.
- A specification discussed in the work item's comment thread is
  deliberation, not the record: the record is the file on the branch when
  the discussion is closed or the comment is posted, and the platform's
  review-approval state is never the record.
- Filling a slot by rewording the base's sentence, or by leaving a
  placeholder for "later", breaks the base's own delivery checks; insert
  whole sentences at the registered structure and grep before inserting.
- A design approved as part of the package is a set of bounds, never a
  procedure; a contract that lets it list steps turns the gate into a
  method review.
- A framework's automation ages with the framework; the moment a tool
  command or script lands in the contract or a slot text, the builder has
  taken on a dependency it cannot verify. Name the framework skill instead.
