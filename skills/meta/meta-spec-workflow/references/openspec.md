# Spec-Anchored Change Workflow: OpenSpec

Read when the selected approach is the spec-anchored change workflow. This
file records what the tool generates and assumes and where it collides
with a harness; it records no command set. The tool has already renamed
its entire slash-command family once — the legacy names still appear in
older articles — so before running or documenting any command, run the CLI
with `--help`, list the commands it installed for this project's agents,
and read its current documentation.

## What it generates

- A single top-level `openspec/` directory holding:
  - `specs/<domain>/spec.md` — the **source of truth** for each domain's
    current behavior, written as requirements with scenarios;
  - `changes/<change-name>/` — one directory per change in flight, with a
    proposal, optional design, tasks, and **delta specs** that state
    added, modified, and removed requirements against the main specs;
  - `changes/archive/` — completed changes, date-prefixed;
  - an optional project configuration file.
- Command or skill files for each supported agent, spelled the way that
  agent loads them; the tool writes them into the agent's own directory.
- A validator (run through the CLI) that checks spec and change structure,
  and an archive operation that merges delta specs into the main specs.

Verify the exact tree from the initializer's output; record it in the
contract's artifact map with the date.

## What it assumes about the repository

- Node.js (a current LTS) for the CLI; git.
- Spec-anchored: main specs describe the system as it is and are updated
  only through archived changes. Editing a main spec by hand outside a
  change breaks the model; say so in the contract.
- Built for existing code: its own guidance says to write specs only for
  what is about to change and warns against back-filling. The contract
  restates that rule as project policy.
- Capability-organized: main specs are per domain or capability, and each
  change amends them. That shape is also the default for a greenfield
  library, framework, or infrastructure project, whose contract is its
  capabilities rather than a sequence of delivered features.
- Requirements are one normative statement each, with scenarios in a
  given / when / then shape; the validator enforces the structure, so the
  project's spec-quality rules should match its format rather than invent
  a second one.
- The tool ignores git branches; branch naming follows the branching
  contract, and the change name and branch name need a recorded mapping
  when they differ.
- Its propose step generates the proposal, delta specs, design, and tasks
  together. The approval gate reviews the proposal and the delta specs
  only; design and tasks are drafts the implementer finishes after
  approval.
- Its documentation supports two archive timings: after merge (its
  recommendation) or inside the pull request. The automated archive mode
  is the former made mechanical through the tool's non-interactive archive
  operation — verify the flag from the CLI's help — and the in-request mode
  is the latter. Under the split shape, approved change records sit under
  the changes directory on the integration branch until their
  implementation lands; a delta written against a domain spec another
  change archived later may no longer apply, so re-validate at
  implementation start.

## Collision points with a harness

| Tool-owned fact | Harness file that tends to restate it | Resolution |
|---|---|---|
| Behavior of a domain (`openspec/specs/…`) | knowledge-base files narrating what the system does, README behavior sections | The spec rules; the knowledge base keeps conventions and mechanics and links the spec |
| Acceptance of a change (delta spec scenarios) | intake templates with acceptance fields, work items | Work items link the change; acceptance is never copied |
| Task list of a change | tracked work status | Work items are created from the task list; status lives in tracked work, completion of the change lives in the archive step |
| Agent command files written by the initializer | an existing project skill covering the same steps | Keep one set; merge or remove the other |
| The optional project configuration | entrypoint statements about spec locations | The configuration rules on layout; the entrypoint points |
| An approved, unimplemented change record on the integration branch (split shape) | tracked work that forgot it | Every such record has an open work item owning its implementation; one without is stale and is assigned or removed through a change request |

## Adopting the tool in an existing repository

Initialize in the repository root after the step-3 snapshot and with the
user's approval; the initializer creates the directory and agent files
and does not touch application code. The tool's onboarding command walks a
first small change end to end — use it, or an equivalent hand-picked
change, as the adoption pilot. Do not create main specs for domains no
change touches; the codebase map covers them.

## Record for the contract

Level (spec-anchored), the verified `openspec/` layout, the rule that main
specs change only through archived changes, the change-name-to-branch
mapping, which agents received command files, whether the validator runs
in the project's checks, the change request shape, the archive mode (and
for automated, the job that runs the archive script and the push
authorization it needs), and the verification date.
