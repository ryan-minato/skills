# Spec-First Kit: GitHub Spec-Kit

Read when the selected approach is the spec-first kit. This file records
what the kit generates and assumes and where it collides with a harness;
it records no command set. Before running or documenting any command, run
the kit's CLI with `--help` and read its current documentation — the
integration flag, the initializer options, and the slash-command names have
all changed between releases.

## What it generates

- A project principles file, the **constitution**, under the kit's own
  hidden directory (`.specify/memory/` at the time of writing). The kit's
  plan step reads it; it is the tool-owned home for engineering principles.
- One numbered directory per feature under a top-level `specs/` directory,
  holding the feature spec, the plan, the task list, and optional research,
  data-model, contract, and quickstart files.
- Templates and helper scripts under the hidden directory.
- Agent-specific command files for the integration chosen at
  initialization — written into that agent's own directory (`.claude/`,
  `.github/`, and so on), one set per integration.

Verify the exact paths from the initializer's output and the tree it
leaves; record them in the contract's artifact map with the date.

## What it assumes about the repository

- Python and the `uv` or `pipx` toolchain for the CLI; git.
- Spec-first: each feature's spec is written before its code, and the
  kit's own guidance does not oblige anyone to keep it current afterwards.
  Choosing spec-anchored with this kit is a project rule the contract must
  state explicitly, because nothing in the kit enforces it.
- The feature script creates a numbered **directory**, and at the time of
  writing no git branch. Any branch must follow the project's branching
  contract; record the mapping from feature directory number to branch
  name, and verify against the current script whether it still creates
  only the directory.
- One constitution per repository. A monorepo with divergent principles per
  package fights the tool; say so before selecting it there.
- Feature-shaped: every spec is a delivery with its requirements and its
  solution. A library, framework, or infrastructure project whose contract
  is a set of capabilities ends up with that contract scattered across
  feature directories; the capability-organized change workflow fits it
  better even with no code yet.

## Collision points with a harness

| Tool-owned fact | Harness file that tends to restate it | Resolution |
|---|---|---|
| Engineering principles (constitution) | `AGENTS.md`, a conventions knowledge file | The constitution rules; the entrypoint carries one pointer; move duplicated principles out of the knowledge base |
| Feature behavior (`specs/NNN-*/spec.md`) | design notes, README feature lists, issues with acceptance criteria | The spec rules; others link it |
| Task breakdown (`tasks.md`) | tracked work items | Work items are created from the task list and link back; the task list is not edited to track status |
| Agent command files written by the initializer | an existing project skill or command set for the same agent | Keep one; a second set of commands for the same steps is a duplicate to remove or merge |
| The kit's template text ("branch name", "NEEDS CLARIFICATION" markers) | project glossary | Leave the kit's vocabulary inside its files; do not import it into the contract |

## Adopting the kit in an existing repository

The kit documents an existing-project path: initialize in place with its
force option, which may overwrite files at paths the kit manages. Do it
only after the snapshot from step 3 of the workflow, on a branch, with the
user's approval, and diff afterwards. The kit's convergence step assesses a
codebase against its artifacts and appends the remaining work; treat that
as the pilot for adoption, and still specify only the features being
changed — the kit does not ask for specs of untouched code, and neither
does the contract.

## Record for the contract

Level (spec-first by default; spec-anchored only with a stated project
rule), the verified constitution path, the `specs/` layout and numbering,
the feature-directory-to-branch rule, which integration's command files
exist, the change request shape (under split, only the feature
specification merges in the specification change request; plan and tasks
join the implementation request), the archive mode — the kit has no
archive operation, so automated archiving exists only if the project
defines a completion criterion (a ticked task list, a status line) and a
post-processing step of its own, recorded here before any job is designed
— and the verification date.
