# Spec: skill-creator

## Purpose

Guides an agent through building a well-structured, high-quality agent skill — covering
both spec compliance and the practices that make a skill effective in production.

The skill serves two mutually reinforcing roles:

1. **Context provision** — supplies spec rules, workflow steps, and instruction patterns
   the agent cannot reliably produce from training knowledge alone.
2. **Output restriction** — constrains agent output via gotchas and a concrete workflow,
   reducing variance in skill quality across runs.

## Trigger Conditions

**Explicit:**
- "create a skill for X"
- "scaffold a skill"
- "write a skill"
- "build a new skill"
- "make a skill"
- "fix this skill"
- "clean up this skill"
- "refactor this skill"
- "update the skill description"
- "add a gotcha to the skill"
- "add a script to the skill"

**Indirect (must also trigger):**
- "I want agents to do X automatically"
- "package this workflow as a skill"
- "how do I teach the agent to do X"
- "help me document this workflow for agents"
- "the agent keeps doing X wrong — I want to fix it"

## What the Agent Lacks

Without this skill, an agent would likely:

- Write skill content for humans (verbose, polite) rather than for agents (concise,
  directive), wasting context budget
- Add too many options instead of one default with a brief alternative, causing
  inconsistent downstream choices
- Write general advice in Gotchas ("handle errors appropriately") instead of
  domain-specific, non-obvious facts
- Create empty optional subdirectories (`scripts/`, `references/`, `assets/`)
  preemptively
- Use non-compliant names (uppercase, underscores, spaces, consecutive hyphens)
- Write a `description` that covers only the explicit domain name, missing indirect
  trigger phrasings
- Load reference files unconditionally, defeating progressive disclosure
- Write templates that over-constrain output by specifying content rather than structure
- Embed repo-internal steps in the generated `SKILL.md` body — for example, "create
  `docs/specs/<name>.md` first" or "run `validate_harness.py`" — making the skill
  unusable when deployed outside this repository where those files do not exist

## Scope

One coherent work unit: creating or modifying a single skill from definition through
a working first draft. Scope includes:

- Planning the skill's purpose, trigger conditions, and necessary content
- Writing `SKILL.md` (frontmatter + body)
- Adding scripts, references, and assets within the skill directory
- Reviewing output against the spec

Out of scope:

- Designing the skill's domain content — the agent is a facilitator; domain knowledge
  must come from the user or external artifacts, not from LLM training knowledge
- Managing multiple skills simultaneously
- Tooling, CI, or repo infrastructure decisions in the consuming project
- Referencing repo-internal files (`docs/specs/`, `WORKFLOW.md`, `QUALITY.md`,
  `scripts/validate_harness.py`) in the generated `SKILL.md` body — the skill must
  be deployable standalone in any project that does not contain this repository

## Content Boundaries

**Include in `SKILL.md` body (loaded on every invocation):**
- Core principles that shape how the agent writes all skill content
- Workflow steps covering the full creation process
- Gotchas — the highest-value content, derived from known failure modes only
- Instruction patterns and guidance on when to use each type

**Never include in `SKILL.md` body:**
- References to files outside `skills/<name>/` — no `docs/specs/`, `WORKFLOW.md`,
  `QUALITY.md`, `scripts/validate_harness.py`, or any other repo-level path.
  End users deploy only the skill directory; those files do not exist in their context.

**Move to `references/` (loaded on demand):**
- Hard spec constraint tables (frontmatter field rules, name character rules, body
  limits) — needed only when validating or uncertain about a constraint

**Move to `assets/` (loaded on demand):**
- Minimal `SKILL.md` frontmatter boilerplate — the truly invariant parts only; no
  sample body content that would constrain the agent's output

## Gotchas Source

All gotchas in `SKILL.md` must come from confirmed failure modes, not generic advice.

1. Name in frontmatter not matching parent directory name → spec validation fails
2. Empty optional subdirectories created preemptively → misleads agents about available
   resources
3. Reference load instructions without precise conditions → agent loads all files on
   every run, defeating progressive disclosure
4. Templates that include sample content (not just structural boilerplate) → agent
   produces variants of the template instead of generating appropriate content
5. `description` covering only explicit domain name → skill fails to activate on valid
   indirect use cases
6. General advice in Gotchas ("validate inputs") → wastes tokens, no signal value
7. Multiple equal options presented → inconsistent tool/approach selection across runs
8. Embedding repo-internal steps in generated `SKILL.md` (e.g. "create
   `docs/specs/<name>.md`", "run `validate_harness.py`") → skill becomes unusable
   when deployed outside this repository where those files and scripts do not exist

## Quality Bar

A skill produced with this skill passes if:

- `name` in frontmatter matches the directory name exactly
- `description` uses imperative phrasing and covers at least three indirect trigger
  phrasings the user might say without naming the domain
- Every reference file load instruction specifies a precise trigger condition
- `SKILL.md` body is under 500 lines
- An agent loading the resulting skill produces correct output for the target task on
  the first attempt without additional user guidance
