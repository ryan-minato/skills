# AGENTS.md

## Repository Purpose

This repository stores reusable agent skills: self-contained instruction sets that an
LLM loads to perform a class of tasks. Skills are authored, reviewed, and maintained here.

## Structure

```
skills/              ← installable skills; flat — one directory per skill name
docs/                ← knowledge base for this repository
  specs/             ← per-skill intent definitions; one file per skill
  QUALITY.md         ← acceptance criteria for skills
  WORKFLOW.md        ← authoring and review process
  REFERENCES.md      ← external documentation links
.agents/skills/      ← symlink → skills/ (allows skills here to load each other)
.github/workflows/   ← CI pipeline definitions (GitHub Actions)
scripts/             ← repo-level validators and formatters
ARCHITECTURE.md      ← structural rules and ownership map
SECURITY.md          ← secrets, PII, and content safety rules
```

## Core Conventions

- **Language:** All docs, code, templates, and comments are English-first.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.
  - Skill changes require a scope: `feat(skill-name): add validation script`
  - Repo-wide changes omit scope: `docs: update QUALITY.md`
- **Scripts:** Python (uv) or Deno only. Both require a shebang and inline dependency
  metadata. See [ARCHITECTURE.md](ARCHITECTURE.md) for the language policy.

## Keeping the Harness Current

Any change that affects repo structure, tooling, CI, or documented conventions **must
update all affected harness files in the same commit**. An outdated or contradictory
harness is worse than none — it actively misleads agents into producing wrong output.

## When to Read What

| Read | When |
|---|---|
| [docs/QUALITY.md](docs/QUALITY.md) | Authoring or reviewing a skill |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Starting or handing off authoring work |
| [docs/specs/](docs/specs/) | Understanding the intent of a specific skill |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Modifying repo structure or ownership boundaries |
| [SECURITY.md](SECURITY.md) | Handling external content, secrets, PII, or script execution |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Looking up external documentation |

## Validation

| Script | Scope | Command |
|---|---|---|
| `validate_harness.py` | Repo structure, file pairing, symlink, AGENTS.md budget | `uv run scripts/validate_harness.py` |
| `check_skill.py` | Single SKILL.md frontmatter and body | `uv run scripts/check_skill.py <path>` |
