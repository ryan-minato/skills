# IDE-Native Spec Format: Kiro

Read when the selected approach is Kiro's native spec format. This file
records what the tool generates and assumes and where it collides with a
harness; it records no command set. Verify current behavior from Kiro's
own documentation before documenting any step — the product spans an IDE
and a CLI, and features differ between surfaces and releases.

## What it generates

- `.kiro/specs/<feature>/` with three files per feature: requirements
  (user stories and acceptance criteria in EARS-style "WHEN … THE SYSTEM
  SHALL …" statements, or a bug-analysis variant for bug-fix specs), a
  technical design, and a task list with trackable items.
- `.kiro/steering/` — project-wide context files (product, tech, and
  structure by default) with per-file inclusion modes: always, matched by
  file pattern, manual, or automatic by description.
- `.kiro/hooks/` — event-triggered actions as JSON files.
- Kiro also reads `AGENTS.md`, at the workspace root and in subdirectories.

## What it assumes about the repository

- The team works inside Kiro. Its specs are plain markdown, so another
  agent can read them, but no first-party statement verified at the time
  of writing promises that other agents honor the three-file workflow or
  the steering inclusion modes. State this as unverified in the contract
  when more than one agent works the repository, and prefer a
  tool-agnostic approach there.
- Spec-first by design: the three files drive one feature's
  implementation. Keeping them current afterwards is a project rule the
  contract must state, because the tool does not oblige it.
- Steering files are the tool-owned home for principles and conventions —
  the same role a constitution plays elsewhere.

## Collision points with a harness

| Tool-owned fact | Harness file that tends to restate it | Resolution |
|---|---|---|
| Principles and conventions (steering) | `AGENTS.md`, knowledge-base convention files | Because Kiro reads `AGENTS.md` too, decide once which file rules each fact; the default is steering for what Kiro applies during specs and `AGENTS.md` for what every agent must see, each pointing to the other, neither restating |
| Feature requirements and acceptance | intake templates, work items | Work items link the spec; acceptance is never copied |
| Task list | tracked work status | Work items are created from the task list; status lives in tracked work |
| Hooks | harness hooks or CI checks for the same event | One owner per event; record which |
| `.kiro/specs/<feature>/design.md` | a root `DESIGN.md` | Different files; the root name is reserved for the visual-design format and must not be created from, or renamed to, the tool's file |

## Adopting the tool in an existing repository

Kiro creates the directories when the first spec or steering file is
written; there is no initializer to snapshot around, but the same
reconciliation applies to every steering file it generates from the
codebase. Specify only the feature being changed next; the codebase map
covers the rest.

## Record for the contract

Level, the `.kiro/` layout, which facts steering rules on versus
`AGENTS.md`, the hook and event ownership, the portability caveat, and
the verification date.
