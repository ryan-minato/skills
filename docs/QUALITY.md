# QUALITY.md — Skill Quality Standards

## What Makes a Good Skill

A skill is good when an agent loading it produces correct, consistent output for the
intended task without additional guidance. Evaluate against that criterion, not against
documentation completeness.

## Content Requirements

**Ground in real artifacts, not training knowledge.**
Base content on: internal runbooks, API specs, code review comments, git diffs, incident
reports. Generic best-practice summaries are low-value — hard to distinguish from the
agent's baseline knowledge.

**Add what the agent lacks; omit what it knows.**
Include: project-specific conventions, non-obvious edge cases, exact tool names and
flags, sequencing constraints that would not be guessed. Exclude: general explanations
of standard concepts (HTTP, CSV, database migrations).

**Scope coherently.**
One skill = one coherent unit of work. Too narrow forces multi-skill loads and risks
conflicting instructions. Too broad makes trigger conditions unreliable.

**Calibrate specificity to fragility.**
Be prescriptive when an operation is fragile or must follow a specific sequence.
Give the agent freedom when multiple valid approaches exist — in those cases, explain
*why* rather than prescribing rigid steps.

**Target general, cross-framework, cross-system audiences.**
Skills in this repository are publicly released for any agent using any framework or
platform. Do not assume a specific language runtime, cloud provider, or proprietary
toolchain. If a skill genuinely requires a specific environment, state the restriction
explicitly in `docs/specs/<name>.md`.

**Document external dependencies in the spec, not the skill body.**
If a skill depends on an external API, library, component, or service, record its name,
version constraints, and access requirements in `docs/specs/<name>.md`. The SKILL.md
body should reference the spec rather than embedding raw API details inline.

## SKILL.md Structural Constraints

| Field | Rule |
|---|---|
| `name` | Lowercase `a-z`, digits `0-9`, hyphens only; max 64 chars; must match directory name |
| `description` | 1–1024 chars; imperative phrasing; covers indirect trigger phrasings |
| `compatibility` | ≤ 500 chars; include only when environment requirements exist |
| Body length | Under 500 lines and under 5,000 tokens |

The `description` is the only text an agent reads before deciding to activate the skill.
Cover indirect phrasings — not only the explicit domain name.

## Body Writing Rules

- Write step-by-step workflow instructions, not a feature list.
- Include a **Gotchas** section: domain-specific facts the agent would get wrong by
  reasonable assumption. Update on every correction. Do not add general advice
  ("handle errors appropriately").
- Reference files with conditional load instructions: specify *when* to load each file,
  not just that the file exists.
- Move content needed only in specific conditions to `references/`; keep `SKILL.md` to
  what is needed on almost every run.
- One default approach per task; mention alternatives in a single sentence. Do not
  present equal options — agents given menus make inconsistent choices.

## Script Rules (Within Skills)

Scripts inside `skills/<name>/scripts/` must:

- Be Python (uv) or Deno only
- Include a shebang and inline dependency metadata
- Accept all input via CLI flags; no interactive prompts
- Output structured data (JSON preferred) to stdout; diagnostics to stderr
- Implement `--help`
- Use exit codes: 0 success, 1 error, 2 bad arguments

If a skill bundles scripts, the `SKILL.md` metadata must briefly note the required
runtime (`uv` or `deno`).

## Spec Requirement

Every `skills/<name>/` must have a corresponding `docs/specs/<name>.md` defining its
purpose and target trigger conditions before the skill is considered complete.

## Review Checklist

- [ ] `name` in frontmatter matches directory name exactly
- [ ] `description` uses imperative phrasing and covers indirect triggers
- [ ] `Gotchas` contains only non-obvious, domain-specific facts
- [ ] Every reference file load instruction specifies a precise condition
- [ ] All scripts implement `--help` and accept input via flags only
- [ ] Skill makes no assumption about framework, runtime, or platform (unless
      `docs/specs/<name>.md` explicitly restricts scope)
- [ ] All referenced external APIs, libraries, or services are documented in
      `docs/specs/<name>.md`
- [ ] No internal business logic, proprietary specs, or confidential runbooks
      (see SECURITY.md — External Content)
- [ ] `docs/specs/<name>.md` exists and matches skill intent
- [ ] `uv run scripts/check_skill.py skills/<name>/` passes without errors
- [ ] No credentials, PII, or verbatim proprietary content
