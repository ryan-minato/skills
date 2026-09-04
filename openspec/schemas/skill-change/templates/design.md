## Context

<!-- Current shape of the skill or tool; binding constraints (catalog CONTEXT.md, dependency range, size limits, mirrors). See proposal.md for motivation. -->

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| <!-- Kind: name --> | <!-- SKILL.md ## Section / references/<file> / assets/<file> / scripts/<file> --> | <!-- the sentence in SKILL.md that loads it --> |

## Description

<!-- For a description change: what it must contain - the capability stated, the situations and phrasings (direct and indirect) that must load the skill, the neighbouring requests that must not, the size budget (1024 limit, warn above 900). Not the wording: that is implementation, tuned against the Trigger scenarios. Delete otherwise. -->

## Dependencies and handoffs

<!-- Every skill named: allowed range, installer routing, fallback when declined. Delete if none. -->

## External impact

<!-- Everything outside the skill's directory the change touches or must keep consistent: other skills, catalog CONTEXT.md and README pairs, symlinks and marketplace, mirrored files, this repository's harness - each with the command or readback that proves it. -->

## Decisions

<!-- Choice, rationale, alternative rejected, requirement served. -->

## Risks / Trade-offs

<!-- [Risk] → Mitigation -->

## Verification plan

<!-- Written before implementation; results go to the pull request's Validation section, never here. -->

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| <!-- domain: scenario --> | <!-- exact prompt, or the outcome task --> | <!-- observable items; mark critical --> | <!-- aggregate score --> | <!-- least capable tier claimed --> | <!-- telemetry or SKILLS_LOADED --> | <!-- candidate worktree or degradation --> |

Script and tool harnesses:
- <!-- scenario: commands run, expected exits -->

Skipped:
- <!-- scenario: reason -->

## Open Questions

<!-- Only what can be settled later without changing specs, placement, or tasks. Delete if none. -->
