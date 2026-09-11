# The Research-Task Convention

Read when depositing the convention, when the project runs OpenSpec, or
when it runs no spec tool.

## The convention

A research task is one objective judged by one evaluation. Its spec
carries an Objective and an Evaluation always, and Context, Search
Scope, Constraints, Completion Condition, Hypotheses as needed; any
section may be one line, and an empty one is deleted, never filled. One
task maps to one pull or merge request that carries the intent, the
source development, the decision, and links to the evidence; hypotheses
and runs live inside it, and metrics live in the tracker. The spec
evolves as findings arrive; run history is never rewritten, and a run
stays attributed to the spec version it ran under. A task closes on its
completion condition — a negative or inconclusive result backed by
evidence is a valid, complete outcome — and archives as the project's
contract says.

Deposit this in `AGENTS.md` (`## Research tasks`) with its reasons, and
record the research task as the project's unit of research work so the
workflow-design and specification-workflow builders that run later read
it as a settled fact.

## Where the spec lives

- The project runs a specification contract → inside it (for OpenSpec,
  a change under the `research-task` schema).
- No spec tool → `research/<task>/spec.md` from
  `assets/research-spec.md`, with the
  hypothesis log beside it.

## OpenSpec projects

Copy `assets/openspec/research-task/`
— `schema.yaml` and `templates/{research,hypotheses,tasks}.md` — to the
tool's project schema directory (`openspec/schemas/research-task/` in
the pinned version; confirm the path and whether the tool's schema
commands can install it by running `openspec --help` and the schema
subcommand's help, never from memory). A research change selects the
schema in its `.openspec.yaml` (`schema: research-task`) and is
spec-less (`skip_specs: true` — the tool writes it when it creates a
change under a schema with no specs artifact; do not add it a second
time): research tasks hold no domain,
while any product code the project ships keeps domains under the
default schema. Record the selection rule and the tool version in
`AGENTS.md`. The specification-workflow builder, when it runs, keeps a
schema the project already runs.

## Approval

Where the project's contract names a gate owner, the gate covers the
objective and the evaluation only — never the hypotheses or the plan; a
change to either goes back through the gate. The research spec is not a
requirements spec: no scenarios, no acceptance criteria beyond the
evaluation.
