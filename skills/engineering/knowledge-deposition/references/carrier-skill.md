# Carrier: Project Skill

Load this when the chosen carrier is a project skill — a recurring,
non-obvious procedure that is fragile, order-sensitive, or branchy. This
covers the minimal skill a deposit produces; for a substantial skill —
bundled scripts, multiple references, distribution beyond this project —
this pairs with `great-skill-writer`. If it is not installed, install it
from https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill great-skill-writer

## Place the skill

Probe in order: the project's already-populated skill directory (its
neighbors are the convention — copy their layout and naming); otherwise the
platform's documented convention for project skills, commands, or workflow
units (`.claude/skills/` and `.agents/skills/` are common homes, as probing
hints rather than mandates). If the platform has no such mechanism, the
procedure becomes a knowledge file instead — switch to the knowledge-file
reference and deposit it there with an event pointer.

## Name it

Kebab-case, named for the procedure's intent — `release-hotfix`,
`regenerate-api-client` — never for the incident that taught it
(`fix-the-june-outage`) or for a file it touches.

## The description is the trigger machinery

Whether the platform announces skill descriptions automatically or an
entrypoint line points here, the description decides when the procedure
fires:

- One third-person capability sentence.
- "Use when …" listing the concrete moments from the finding's trigger
  field, including indirect phrasings a user would say without naming the
  procedure.
- A "Not for …" exclusion when an adjacent task shares vocabulary and must
  not fire it.

If the platform does not announce descriptions, add an event-triggered
pointer line to the entrypoint exactly as for a knowledge file.

## Shape the body

- The steps every run needs, in order, and nothing speculative.
- Exact commands where the procedure is fragile or order-sensitive; goal
  plus reason where several ways are valid.
- One recommended default per decision, with at most one sentence on when
  to deviate.
- A gotchas section built from the failure evidence — the non-obvious
  facts an agent would get wrong by reasonable assumption are exactly what
  the finding recorded.
- Update triggers inside the skill: name the command, path, or convention
  change that makes it stale, so a future agent revises instead of
  trusting.

## Keep it minimal

- One skill, one procedure. A second, unrelated lesson gets its own
  deposit; merge only when the two always fire together.
- No `references/` or `scripts/` subdirectories at birth unless a real
  file goes in on day one — a single lesson rarely justifies progressive
  disclosure, and empty directories mislead.
