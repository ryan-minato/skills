# Multi-Agent Topology

Read when the user asks for, or the plan is considering, more than one
agent. Default to a single agent and record what would trigger
reconsideration.

## Why Caution First

- **Framework variance.** Agent frameworks differ in what their
  multi-agent support actually delivers: sub-agent concurrency, context
  sharing, and structured returns vary widely, and a team member on a
  framework without the needed capability runs the whole workflow in a
  degraded mode.
- **Cost blow-up.** Multi-agent runs multiply context. When sub-agent
  prompts do not share stable prefixes, cache misses dominate and cost
  scales far worse than the agent count suggests.

## If The Project Adopts It, Design It Like This

1. **Verify before committing.** Check the target framework's actual
   capability boundary (concurrent sub-agents, context sharing, structured
   returns) and confirm every team member's tooling supports it. If anyone
   would run degraded, stay single-agent.
2. **Design for cache economics.** Keep shared prompt prefixes stable,
   bound fan-out and per-agent context, set a token budget with a hard
   cap, and avoid deeply nested delegation.
3. **Split roles along harness seams.** Separate reading/research,
   building, and verification. One writer at a time; parallel file
   mutation only with isolated worktrees.
4. **Keep orchestration deterministic.** Prefer scripted or
   workflow-defined sequencing over free-form model-driven delegation, and
   document a single-agent fallback path that still works.
5. **Record the decision.** The topology, its rationale, the fallback, and
   the reconsideration trigger all go into the harness plan.
