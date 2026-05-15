# Skill Spec: Hard Rules

## `SKILL.md` Structure

```
---
<frontmatter fields>
---

<Markdown body>
```

## Required Fields

### `name`

| Constraint | Rule |
|---|---|
| Characters | Lowercase `a-z`, digits `0-9`, hyphens `-` only |
| Leading/trailing hyphen | Not allowed |
| Consecutive hyphens | Not allowed (`pdf--tools` is invalid) |
| Max length | 64 characters |
| Directory match | Must exactly equal the parent directory name |

Valid: `pdf-processing`, `data-analysis-v2`, `code-review`

Invalid: `PDF-Processing` (uppercase), `-pdf` (leading hyphen), `pdf--processing`
(consecutive hyphens), `pdf_processing` (underscore), `pdf processing` (space)

### `description`

| Constraint | Rule |
|---|---|
| Required | Yes |
| Min length | 1 character |
| Max length | 1024 characters |

The description is the **only** text an agent reads before deciding to activate the
skill. Write in third person: first sentence = capability (action verbs + domain
keywords); remaining sentences = trigger conditions including indirect phrasings.

## Optional Fields

### `compatibility`

Max 500 characters. Include only when the skill has specific environment requirements
(runtime version, required system packages, network access). Omit otherwise.

### `license`

Short license name or path to a bundled license file. No length constraint.

### `metadata`

Flat key-value map (`string → string`). Use reasonably unique key names to avoid
conflicts across skills.

### `allowed-tools`

Space-separated string of pre-approved tool names. **Experimental** — support varies
by agent implementation.

## Directory Structure

```
skill-name/            ← must equal the `name` frontmatter field
├── SKILL.md           ← required
├── scripts/           ← optional
├── references/        ← optional
└── assets/            ← optional
```

Do not create optional subdirectories unless at least one file will go in them.

## Body Content Rules

- No mandatory format restrictions
- Recommended limit: under **500 lines** and **5,000 tokens**
- File references must use relative paths from the skill root
- Prefer references no more than one directory level deep from `SKILL.md`

## Progressive Disclosure

| Stage | Content | Recommended size |
|---|---|---|
| Metadata | `name` + `description` only | ~100 tokens |
| Instructions | Full `SKILL.md` body | < 5,000 tokens |
| Resources | `scripts/`, `references/`, `assets/` | Loaded on demand |

Specify *when* to load each reference file. Loading all files unconditionally defeats
progressive disclosure and wastes context budget.
