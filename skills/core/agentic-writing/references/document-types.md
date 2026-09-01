# Document types

Read the section matching the document at hand. Every type shares four
requirements: facts and instructions first, the source of truth named for each
fact, completion boundaries the agent can check, and no unresolved
placeholders.

## Entrypoint (AGENTS.md, CLAUDE.md)

Loaded on every task, which makes it the most expensive file in the project.
It is the map for progressive loading, not the manual.

It carries what the project is for, the constraints that hold on every task
(safety rules among them), the commands that validate work, and the exact
conditions for reading anything deeper. Everything only some tasks need moves
behind a pointer; rules that apply to one subtree move into a nested
entrypoint beside that code.

Aim for roughly 100 lines. A small project whose entrypoint is the whole
harness may run longer, and safety rules stay visible whatever the budget.

Name the events that make the entrypoint stale and what each one obliges —
"a public skill added: update the symlink, both catalog READMEs, the plugin
manifest" — so the change that breaks the document also repairs it.

When a host reads its own filename (CLAUDE.md, GEMINI.md), keep one file
authoritative and let the others point at it rather than copying it.

## Knowledge-base entry

One topic per file, sized to load whole. It opens by stating when it applies,
matching the pointer that reaches it, and names the source of truth for what
it asserts.

It records what lookup cannot recover: the convention nobody wrote down, the
reason a choice beat its alternative, the trap a setting hides. Facts one
command answers stay in the environment.

An entry no pointer names is unreachable — add the pointer where the work
happens, or drop the entry.

## Agent-facing spec format (DESIGN.md and the like)

A file whose name is a contract: a tool or convention treats it as
authoritative for one domain. Declare at the top what it governs and the
format it uses.

Keep machine-checkable content in structured blocks a script can parse and
validate, and reserve prose for what a checker cannot express. Where a
generator or validator exists, name it and the command that runs it.

The name is reserved for the format — repurposing it for an ordinary document
breaks every agent that expects the contract.

## Prompt or instruction file

State the role and the goal before any detail, then ordered steps with a
checkable done condition on each step that can be left half-finished. Give the
output format explicitly when the result feeds another tool: a shape stated
once beats a shape corrected every run.

Inputs the caller substitutes appear as marked placeholders with a stated
expected value, so an unfilled slot is visible rather than silently shipped.
