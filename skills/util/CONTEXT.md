# util — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

Skills here act on the working process itself — how the agent and the user
clarify, question, decide, and organize ideas together. Their product is a
shared understanding, not code, prose, or project files.

## Requirements

- Process only, domain-free: the instructions must hold whatever the user is
  working on. A skill that only makes sense for code, for a document, or for
  one tech stack belongs in another catalog.
- Leave no files in the user's workspace: the working material is the
  conversation. When an artifact is genuinely required, the skill directs it
  to the harness scratchpad or the platform temporary directory, so a stray
  draft cannot be committed by accident.
- Degrade across harnesses: prefer the host's own capabilities — structured
  question tools, subagents — and state an explicit fallback for hosts that
  lack them. Never require a specific framework, tool name, or API.
- Opt-in rather than global: these skills change how the agent conducts a
  conversation, which is a personal or team preference. Default install scope
  is per project, unlike `core`.

## Disambiguation

Clarifying, stress-testing, or pinning down a plan, an idea, or a decision
that is already on the table → `clarify-thinking` · defining what something
should achieve from scratch and recording it as a goal document →
`goal-alignment` in `engineering` · resolving ambiguous requirements while
already writing code → `programming-guidelines` in `core` · producing the
implementation plan or task breakdown itself → none of them (out of catalog
scope).

## References

_(none yet — add catalog-scoped reference URLs here)_
