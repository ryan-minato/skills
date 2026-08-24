# When A Skill Is Warranted

Read when it is unclear whether a procedure deserves a skill. The cheaper
carriers — an entrypoint line, a knowledge file — win unless the procedure
clears the bar.

## The Bar

A skill is warranted when the procedure is all of:

- **recurring** — it will run again, in future sessions, by agents that
  were not present when it was learned;
- **non-obvious** — an agent without the skill gets it wrong or asks; and
- at least one of: **fragile** (small mistakes have real cost),
  **order-sensitive** (steps must run in sequence), or **branchy** (the
  right action depends on conditions worth encoding).

## Cheaper Carriers

| Content | Carrier |
|---|---|
| a one-sentence rule ("always run X before Y") | entrypoint line |
| stable facts with no procedure (versions, limits, domain truths) | knowledge file |
| context needed once, this session | nothing — say it, don't store it |
| a procedure that will change before it runs again | nothing yet; wait for it to stabilize |

## Anti-Patterns

- A skill wrapping a single command that the entrypoint's command table
  already lists.
- A "general guidelines" skill — no trigger can fire precisely, so it
  either always loads or never does.
- A skill created from one incident. One occurrence is an anecdote;
  encode it after it recurs.
- Splitting one procedure across two skills so each is "smaller" — the
  seam becomes a failure point.
