# Advanced Autonomy

Read this when the user asks for a self-evolving harness, persistent memory,
unattended operation, autonomous task routing, failure-driven knowledge
updates, or multiple agent roles, or when project evidence makes one of these
independent choices worth proposing. Treat every precondition below as missing
until verified. When a precondition is missing, retain supervised operation or
a single-agent topology and document what must change before revisiting it.

## Preconditions for self-evolution or unattended operation

- Isolation and approval boundaries that make unattended operation safe.
- Durable memory in a location agents can read and write across sessions.
- Validation strong enough to catch harness drift, not just code defects.
- Explicit rules for what agents may change without review, with approval
  and rollback paths for everything else.
- An audit mechanism: what changed, why, triggered by what.

Ask the user before granting agents authority to change their own harness
or write to external systems; these are decisions no evidence in the
repository can substitute for.

## Failure-driven feedback loops

Agent failures may feed harness improvement only under these controls:

- A failed run produces findings first; nothing edits guidance directly
  from a failure.
- Each proposed update cites the failure pattern, the checked source of
  truth, and how the fix was verified.
- Updates land through the approval path above, and the audit trail records
  them.

## Preconditions for multiple agent roles

- The agent framework actually supports coordinated multiple roles.
- The project is complex enough that separated contexts beat one shared
  context; splitting roles reduces cache hits and multiplies token use, so
  the isolation must earn its cost.
- Role boundaries are clear and non-overlapping — divide by context needs,
  not by mirroring the human org chart.
- Review of cross-role output is defined and, unless the user decides
  otherwise, human-owned.

## What the project harness must record

Advanced autonomy cannot depend on this skill being available. The project's
own harness must state:

- The autonomy boundaries: what agents decide, what needs approval, which
  workflow actions agents may initiate.
- Where durable memory lives and how it is read and written.
- How tasks are routed, validated, and audited.
- How self-maintenance changes are reviewed and rolled back.
- How to stop or downgrade the autonomy — a visible fallback path to
  supervised operation.

## Plan output

An advanced-autonomy plan states, beyond the main plan in SKILL.md: each
requested autonomy capability and why supervised single-agent operation is
insufficient, the isolation and approval model, the durable memory design,
role responsibilities, the
validation-and-audit design, the feedback-loop rules, the delegated
workflow actions, and the rollback and human-review points.
