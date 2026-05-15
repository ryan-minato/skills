# ARCHITECTURE.md

## Directory Responsibilities

```
skills/              ← installable skills; flat — one directory per skill name
docs/                ← knowledge base for this repository
  specs/             ← per-skill intent; one file per skill (docs/specs/<name>.md)
.github/
  workflows/         ← CI pipeline definitions (GitHub Actions)
scripts/             ← repo-level validators and formatters (Python/uv or Deno only)
.agents/skills/      ← symlink → ../skills; allows skills here to load each other
AGENTS.md            ← agent entry point; map only (target 50–100 lines, hard limit 120)
ARCHITECTURE.md      ← this file
SECURITY.md          ← security constraints
```

## Key Invariants

**Skill–spec pairing:** For every directory `skills/<name>/`, a file
`docs/specs/<name>.md` must exist. Names must match exactly. Create the spec file
before or alongside the skill directory.

**Flat skills layout:** `skills/` contains only top-level skill directories. No skill
nested inside another skill directory. Each skill directory must contain `SKILL.md`.

**Symlink target:** `.agents/skills` is a symlink to `../skills`. Do not add files here
directly; modify `skills/` instead.

## Skill Directory Structure

Defined by the skills specification (accessible via the `skills-spec` MCP server in
the devcontainer):

```
skills/<name>/
├── SKILL.md          ← required
├── scripts/          ← optional; skill-level scripts (Python/uv or Deno)
├── references/       ← optional; on-demand reference material
└── assets/           ← optional; templates and static resources
```

## Script Language Policy

All scripts in `scripts/` (repo-level) and `skills/<name>/scripts/` (skill-level) must:

- Be Python (uv) or Deno — no other runtimes
- Include a shebang line
- Include inline dependency metadata (`# /// script` block for uv; locked import
  specifiers for Deno)
- Accept all input via CLI flags; no interactive prompts

If a skill bundles scripts, its `SKILL.md` metadata must briefly note the required
runtime (`uv` or `deno`).

## Skill Naming Rules

| Constraint | Rule |
|---|---|
| Characters | Lowercase `a-z`, digits `0-9`, hyphens `-` only |
| Leading/trailing hyphen | Not allowed |
| Consecutive hyphens | Not allowed (`pdf--tools` is invalid) |
| Max length | 64 characters |
| Directory match | `name` in `SKILL.md` frontmatter must equal the parent directory name |

## AGENTS.md Size Budget

Target: 50–100 lines. Hard limit: 120 lines. Exceeding the limit means content should
move to `docs/` with a pointer added to AGENTS.md. Enforced by
`scripts/validate_harness.py` (structural validator).

## Deployment Boundary

`skills/<name>/` is the deployable unit. When a skill is installed in another project,
only the contents of that directory are available — no `docs/`, `scripts/`, `AGENTS.md`,
`WORKFLOW.md`, `QUALITY.md`, or any other file from this repository.

**Consequence for skill authors:** `SKILL.md` body and any skill-local files must be
fully self-contained. Never instruct an agent to read or reference files outside
`skills/<name>/`.

Prohibited in `SKILL.md` body:
- Paths to `docs/`, `scripts/`, or any repo-level file
- References to `docs/specs/<name>.md`, `WORKFLOW.md`, `QUALITY.md`, or `SECURITY.md`
- Instructions to run repo-level scripts (`validate_harness.py`, `check_skill.py`)
