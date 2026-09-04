---
name: meta-spec-workflow
description: >-
  Disposable builder skill (delete after the harness is built): aligns a
  project's harness and work tracking with spec-driven development — owns
  the target-constraints layer: specifications, their tooling, and the rule
  that tracked work links specifications, not restating them. Settles
  the specification level and tool with the user, adopts the tool's
  layout, gives every fact one source of truth between tool-owned files
  and AGENTS.md or the knowledge base, and deposits a specification
  contract written in the project's platform vocabulary for the platform
  builders to implement. Use in a harness build when the
  workflow contract records spec-driven intent, when a spec tool's
  directories sit beside an agent harness, or when asked to make the
  harness, templates, or tracker match Spec-Kit, OpenSpec, or Kiro, or to
  stop issues and specs contradicting each other. Not for writing specs,
  choosing the practice, the spec loop, the tracker, or branching.
license: Apache-2.0
compatibility: The bundled detection script requires Python 3.10+ (stdlib only).
---

# Specification Workflow Contract

Settle with the human developers how this project writes, approves, and
keeps its specifications, make the harness say the same thing the chosen
tool assumes, and leave the result in the target project as a contract the
platform lifecycle builders implement. The questions and the design are
settled in tool and workflow terms with no hosting platform in mind; the
deposited contract names the platform's objects — issues or work items,
pull or merge requests, drafts, the automation that archives — because the
agents that read it work on that platform. It must survive this builder and
this conversation.

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
- Tool commands are verified, never recalled. Every reference here names
  what a tool generates and assumes, not its current command set; run the
  tool's own `--help` and read its current documentation before invoking
  or documenting a command.
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
anything else, because no contract is deposited without it. Sort everything into known facts with evidence, unknown facts
still discoverable, and decisions only a human can make.

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
   user asks to change it.
3. **Approval gate** — who approves a specification before planning and
   implementation start, and whether an agent may approve its own. Record
   the owner; the authority builder attaches its levels to this gate later.
4. **Division of labor** — confirm the default: specifications own what,
   why, and acceptance; tracked work owns who, when, and status and links
   the specification. Any deviation is recorded with its reason.

Done when: level, approach, approval owner, and division of labor are
settled by the user or confirmed from evidence, each with its selecting
fact recorded.

### 3. Install or adopt the tool

Read the one reference matching the selected approach:
[spec-kit.md](references/spec-kit.md), [openspec.md](references/openspec.md),
[kiro.md](references/kiro.md), or
[committed-documents.md](references/committed-documents.md) for both the
no-tool and the custom-layout choices.

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
on a behavior, who approves a spec, and how a work item links a spec, from
target-project files alone. Confirm the deposited file does not carry this
skill's disposable marker, name, or paths.

Then hand off by name, in order. Governance next: attach the approval gate
and authority levels with `meta-agent-authority`. Platform expression after
that: run the lifecycle builder for the evidenced host platform, which
consumes this contract to shape intake templates, change-request templates,
and the project workflow skill. If either is not installed, load the
`ryan-minato-skills-installing` skill and install the whole `meta` catalog
at project scope as it directs — its builders stack and are disposed
together; never run an install command yourself.

If the user declines, record in the hand-off which decisions remain
unexpressed on the platform. Every hand-off report ends with the order
above and with the closing step that follows.

When this builder runs under `meta-harness-building`, return there for the
closing step. When it runs alone, once the deposit is verified and before
the work goes to review, ask the user whether to delete the disposable
builders now — the build request is not deletion consent — and on that
decision load `meta-disposal`, which lists, confirms, and removes them. If
the user declines, leave the builders in place and out of every commit, and
record it in the handoff.

## Gotchas

- Spec tools rename commands and move files between releases; one of the
  tools covered here has already renamed its whole command set once. The
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
  common contradiction source; the platform builder must make the template
  link the spec's scenarios instead.
