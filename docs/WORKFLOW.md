# WORKFLOW.md — Skill Authoring Process

> **Contributor scope:** This document describes the authoring process for this
> repository. These steps are **not** skill content — do not embed references to
> `docs/specs/`, `validate_harness.py`, or any other repo-internal file in a
> `SKILL.md` body. See [ARCHITECTURE.md](../ARCHITECTURE.md#deployment-boundary)
> for the deployment boundary rule.

## Authoring a New Skill

### 1. Define Intent

Create `docs/specs/<name>.md` first. It must answer:

- What task does this skill cover?
- What user intents and phrasings trigger it (including indirect ones)?
- What does the agent lack that makes a skill necessary?
- Known constraints or edge cases.
- If the skill involves external APIs, libraries, components, or services: their names,
  version constraints, and access requirements. Do not leave this to the SKILL.md body.

Do not create the skill directory until the spec exists.

### 2. Name the Skill

Verify the name before creating the directory:

- Lowercase `a-z`, digits `0-9`, hyphens `-` only
- No leading, trailing, or consecutive hyphens
- Max 64 characters
- Must match the `name` field in `SKILL.md` frontmatter exactly

### 3. Create the Skill Directory

```
skills/<name>/
└── SKILL.md
```

Use the skills spec frontmatter structure. The `skills-spec` MCP server (available in
the devcontainer) is the authoritative reference for frontmatter fields and constraints.

Create optional subdirectories (`scripts/`, `references/`, `assets/`) only when there
is at least one file to place in them.

### 4. Write SKILL.md Body

- Start with a step-by-step workflow, not a feature list.
- Add a **Gotchas** section early — this is the highest-value content.
- Ground content in real artifacts (runbooks, API specs, git history), not generic
  summaries.
- Use conditional load instructions for reference files: specify *when* to load each
  file, not just that it exists.

See `docs/QUALITY.md` for the full content requirements and review checklist.

### 5. Add Scripts (If Needed)

Add scripts to `skills/<name>/scripts/` only when:

- The logic is deterministic and would be re-generated identically on every run.
- Errors need structured handling that inline generation cannot reliably provide.

Language: Python (uv) or Deno. Both require a shebang and inline dependency metadata.
See `docs/REFERENCES.md` for runtime documentation links.

### 6. Validate

Run both validators before committing:

```sh
# Repo structure: required files, spec↔skill pairing, symlink, AGENTS.md budget
uv run scripts/validate_harness.py

# SKILL.md content: frontmatter keys, description length, body line count
uv run scripts/check_skill.py skills/<name>/
```

Fix any errors before committing. Warnings are informational.

### 7. Commit

Conventional Commits format. Skill-specific changes require a scope:

```
feat(<skill-name>): initial SKILL.md
feat(<skill-name>): add validation script
fix(<skill-name>): correct gotcha about X
```

Repo-wide changes (README, harness docs, CI) omit the scope:

```
docs: update QUALITY.md review checklist
ci: add harness validator to pre-commit
```

## Modifying an Existing Skill

1. Re-read `docs/specs/<name>.md` to confirm intent before editing. *(Contributor
   step — not skill content.)*
2. Make the smallest change that addresses the issue.
3. Update `docs/specs/<name>.md` if the skill's scope changed. *(Contributor
   step — not skill content.)*
4. Add a new Gotcha if the change corrects an agent behavior error.
5. Run the validator.

## Reviewing a Skill

Apply the review checklist in `docs/QUALITY.md`. Reject if:

- `docs/specs/<name>.md` does not exist or conflicts with the skill content.
- `Gotchas` contains only general advice rather than domain-specific facts.
- Scripts have interactive prompts or missing `--help`.
- Credentials, PII, or verbatim proprietary content is present.
