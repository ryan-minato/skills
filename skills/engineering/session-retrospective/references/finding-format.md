# Finding Format

Load this when drafting the findings list. One record per finding; every
field filled before the list is presented.

## The record

```text
Finding <n>: <one-line title>
  signal:      repeated-failure | tool-flagged | expensive-discovery |
               experiment-verdict | overruled-default | docs-mismatch
  evidence:    <occurrence count>; one line per occurrence
  rule:        <the lesson as a standalone instruction>
  trigger:     <the future moment an agent needs this>
  recurrence:  low | medium | high — <one-line reason>
  impact:      low | medium | high — <what it cost this session>
  size:        one line | one file | one skill
  destination: entrypoint line | knowledge file | project skill | none yet
  verdict:     (left blank — the user fills keep / edit / drop)
```

## Field guidance

- **evidence** — an occurrence is one concrete moment in the session: a
  failed command, a user correction, a search that finally landed. The
  user saying "this keeps happening" is one occurrence here plus testimony
  of prior-session recurrence; say so in the line. Describe each
  occurrence in the session's own words; never invent line numbers or
  quotes — evidence the user cannot verify undermines the verdict.
- **rule** — imperative, self-contained, executable by an agent that never
  saw this session. "Run the schema migration before seeding, because the
  seeder validates against the live schema" — not "avoid the problem we
  hit with the seeder". Keep the reason as one clause when the rule leaves
  judgment room.
- **trigger** — the event that makes the rule relevant: "before running
  integration tests", "when adding a database column", "when a request to
  the payments API returns 403". This field becomes the load condition or
  description trigger at deposit time, so name a real, recognizable moment.
- **recurrence** — low: plausible but tied to no scheduled work; medium:
  tied to a task type the project performs regularly; high: fires on every
  task of a named kind.
- **impact** — what the miss cost this session (wasted time, broken
  output, a user correction) as the proxy for what the next miss costs.
- **size** — a size guess only; the destination decision happens at
  deposit time, after probing the project, and may override it.

## Worked micro-examples

- repeated-failure — evidence: `pytest` failed 3× on a stale fixture
  cache; rule: "Delete `.pytest_cache/` after switching branches, because
  fixtures are branch-specific"; trigger: after a branch switch.
- tool-flagged — evidence: formatter rewrapped imports in 4 commits; rule:
  "Let the formatter own import order; never hand-sort"; trigger: when
  editing imports.
- expensive-discovery — evidence: rate-limit config found after searching
  five modules; rule: "Rate limits live in `gateway/limits.yaml`, not in
  service code"; trigger: when changing rate limiting.
- experiment-verdict — evidence: tried streaming and batch export, batch
  won on memory; rule: "Export via the batch path; streaming exhausts
  memory beyond 1M rows"; trigger: when adding an export.
- overruled-default — evidence: agent added retry wrappers, user removed
  them; rule: "Do not add retry logic; the platform gateway already
  retries, and stacked retries multiply load"; trigger: when handling
  transient errors.
- docs-mismatch — evidence: README says `make test`, target was renamed;
  rule: "Tests run with `make check`; the README's `make test` is stale —
  fix the README rather than trusting it"; trigger: before running tests.

## Counter-examples — findings that fail the bar

- A typo fixed once, never seen again: no recurrence, negligible impact —
  not a finding.
- "The CLI supports `--verbose`" when `--help` says so: the environment
  answers this on demand and never goes stale; recording it adds rent
  without value.

## Presentation

Rank by recurrence times impact, descending; break ties toward the smaller
size. Present as a numbered list — each entry showing title, rule,
evidence summary, and recommended destination — and request keep / edit /
drop per finding. Recommend exactly one destination per finding; a hedged
list of options pushes the decision back onto the user.
