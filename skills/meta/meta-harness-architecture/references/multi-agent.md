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
   building, and verification. Read-only roles (review, analysis, scanning)
   share one workspace; parallel file mutation only with isolated
   worktrees or containers.
4. **Namespace every shared resource.** Before running writers in
   parallel, list what they would collide on — local ports, test database
   instances or schemas, fixed-path lock files, build caches, external
   rate limits — and allocate dynamically or per instance.
5. **Keep remote state to one coordinator.** Worktrees do not isolate the
   platform: two agents commenting on one issue are unreadable, two
   editing one request's labels overwrite each other, two pushing one
   branch lose commits. Each agent owns its own remote branch; every write
   to a platform object (issue, comment, label, status) goes through one
   coordinator, human or designated agent, and each such write is
   pre-authorized under the authorization boundary — never decided by a
   parallel agent on its own.
6. **Plan the merge.** Isolation must be undone: name who merges the
   results and who settles conflicts, and split the parallel work along
   file boundaries at planning time, since the split decides how much the
   parallelism pays.
7. **Keep orchestration deterministic.** Prefer scripted or
   workflow-defined sequencing over free-form model-driven delegation, and
   document a single-agent fallback path that still works.
8. **Record the decision.** The topology, its rationale, the fallback, and
   the reconsideration trigger all go into the harness plan.
