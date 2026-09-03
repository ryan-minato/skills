# Skill Quality Standards

Quality bar for every skill in this repository. Read this before creating or
modifying a skill; the `skill-authoring` project skill walks through the
workflow that applies these standards.

## Scoping a skill

- Add what the agent lacks; omit what it knows. Project-specific
  conventions, non-obvious procedures, and edge cases belong in a skill;
  general knowledge (what a PDF is, how git works) does not. If the agent
  already does the task well without the skill, don't build the skill.
- Scope one skill to one coherent unit of work. Too narrow, and several
  skills load together with conflicting instructions; too broad, and the
  description cannot trigger reliably. Split or merge until the boundary is
  clean.
- Teach the approach, not the instance: instructions should generalize
  across variations of the task, not encode one worked example.

## Structure

- A skill is one directory whose only required file is `SKILL.md`. Optional
  subdirectories: `scripts/` (executable code), `references/` (detailed
  documentation loaded on demand), `assets/` (templates, resources). Create
  a subdirectory only when a file is going into it — an empty directory
  misleads agents into expecting files that don't exist.
- **No `README.md` inside a skill root.** The catalog README describes the
  catalog's skills; SKILL.md is the skill's own document.
- Everything is written in English.

## Naming (recommended default)

Names are `[<prefix>-]<body>[-<suffix>]` in lowercase kebab-case. This is
the recommended shape, not a rule: only the spec limits below and the
catalog name prefixes are enforced, a name that departs from the shape is
not a defect, and the name chosen when a skill was created is respected. A
catalog's `CONTEXT.md` may add to or override it in a `## Naming` section.

- **Body** (required): by default `<subject>-<action>` — what the skill
  acts on, then what it does: a singular noun (`git`, `latex`, `goal`,
  `session`) followed by a gerund or action noun (`commit`, `writing`,
  `check`, `alignment`, `refactoring`, `setup`). When the subject is itself
  a proper noun for a standard, format, product, or established concept,
  that noun is the whole body, unsplit and with its own number:
  `conventional-commits`, `gitmoji`, `design-md`, `meta-harness`. One to
  four words. A default name is the body alone.
- **Prefix** (off by default): what the skill belongs to. A catalog prefix
  is reserved by the catalog's `CONTEXT.md` and listed in
  `CATALOG_NAME_PREFIXES` (`meta-`, `scaffold-`), which the validator
  enforces for that catalog's members; an ownership prefix marks a skill
  bound to one named owner or product (`ryan-minato-skills-installing`). A
  proper-noun body that happens to start with a reserved prefix
  (`meta-harness`) is not using the prefix.
- **Suffix** (off by default): the kind of skill, for catalogs where
  several skills share a subject and need telling apart; the vocabulary
  belongs to the catalog's `CONTEXT.md` (`writing`: genre skills end in
  `-writing`, source-medium skills in `-authoring`).
- Prefer to avoid agent nouns (`-writer`, `-helper`), leading verbs
  (`clarify-`), and splitting a proper noun into subject and action.

## Spec limits (mechanically enforced)

`just check-skill <dir>` (and `just validate` across the repo) enforces
these, so a one-line summary suffices here:

| Field | Rule | Level |
|---|---|---|
| `name` | lowercase `a-z`/`0-9`/single hyphens; ≤64 chars; equals the directory name | error |
| `description` | required; ≤1024 chars | error (warn >900) |
| `compatibility` | ≤500 chars; include only for real environment requirements | error |
| Body | recommended <500 lines / ~5,000 tokens | warning |
| Empty `scripts/`/`references/`/`assets/` | don't create them | warning |

## Self-containment and dependencies (public skills)

Users install a skill by copying its directory out of this repository, so a
public skill loses access to everything outside its own root the moment it
is installed, and it cannot assume any other skill was installed beside it.

- No links or path references to repository files, other skills, or anything
  above the skill's root directory. Relative paths inside SKILL.md must stay
  within the skill directory.
- A *dependency* is naming another skill and instructing the agent to load,
  pair with, use, or install it. Describing a role without a name ("a genre
  writing skill"), naming a convention or tool that happens to share a
  skill's name (the gitmoji convention), and provenance URLs in frontmatter
  `metadata.references` are not dependencies.
- Allowed range: every public skill may depend on skills in the `core`
  catalog, which users install globally. Dependencies inside the skill's
  own catalog or on another catalog are allowed only when the catalog's
  `CONTEXT.md` grants them in its `## Dependencies` section (today: `meta`
  builders on one another, `scaffold` builders on `meta`). Default: not
  granted. An in-range dependency may be hard — "load it alongside; do not
  work from this skill alone".
- A skill outside the allowed range may only be an *optional handoff*: the
  skill's own path works without it, the handoff states what happens when
  the user declines, and the description refers to it by role, not by skill
  name.
- No dependency on, and no recommendation of, skills from other
  repositories or skill libraries, unless the catalog's `CONTEXT.md`
  exempts the catalog or a named skill. This library stays independent.
- Never print an install command. Every handoff — in range or optional —
  routes through the `ryan-minato-skills-installing` skill in `core`, which
  owns runner choice, install scope, and whole-catalog installs:

  ```markdown
  This skill pairs with `other-skill`. If it is not installed, load the
  `ryan-minato-skills-installing` skill and install `other-skill` as it
  directs; never run an install command yourself. (If that installer skill
  is absent too, it lives in the `core` catalog of
  https://github.com/ryan-minato/skills.)
  ```

  For a whole catalog: "install the whole `meta` catalog at project scope
  as it directs — its builders stack and are disposed together." For an
  optional handoff, end with "If the user declines, <what this skill does
  instead>."
- Exemption: project-only workflow skills in `.agents/skills/` exist to
  operate this repository, are never distributed, and may reference repo
  paths and each other freely. All other rules in this document still
  apply to them.

## Writing the description

The description is the only thing an agent sees before deciding to load the
skill, so it carries the trigger logic:

- State what the skill does *and* when to use it. Write the capability
  sentence in the third person; write triggers as "Use when ...".
- Include the concrete trigger phrases a user would actually say — including
  indirect phrasings that don't name the domain ("make this readable" for a
  formatting skill).
- Be specific enough to avoid firing on unrelated tasks; a description that
  triggers everywhere is as bad as one that never triggers.

## Writing the body

- Keep SKILL.md lean: it should hold what is needed on most runs, with
  branch-specific material pushed to `references/` (see progressive
  disclosure below) and fragile command sequences to `scripts/`.
- Write concise, directive, complete sentences — no pleasantries, but no
  telegraphic fragments either. An agent that understands *why* an
  instruction exists makes better context-dependent decisions, so explain
  the reason wherever you grant freedom.
- Match specificity to fragility: where multiple approaches are valid, give
  the goal and the reasoning; where operations are fragile, order-sensitive,
  or must be consistent, give exact prescriptive steps.
- For each decision the skill covers, give one recommended default plus, at
  most, one sentence on when to deviate. Presenting equal alternatives makes
  agents choose inconsistently across runs.

## Instruction patterns

Common patterns that have proven useful — not an exhaustive list and not
required sections. Apply whichever fit the skill, and use other structures
when they serve the task better.

- **Gotchas** — a valuable pattern for non-obvious facts the agent would get
  wrong by reasonable assumption (naming inconsistencies, endpoints that
  lie, required flags). Keep entries domain-specific; add one when a
  correction reveals a reusable failure mode. General advice ("validate
  inputs") adds no signal.
- **Templates / assets** — a deliberate trade-off: they constrain output
  variance. Use them exactly when that is the goal (fixed output formats,
  stable boilerplate); avoid them where output needs task-specific judgment.
- **Checklists** — for multi-step workflows with dependencies or validation
  gates.
- **Validation loops** — do → validate → fix → repeat, backed by a check
  that explains its own failures well enough for the agent to self-correct.
- **Plan-validate-execute** — for batch or destructive operations: produce a
  plan, validate it against a source of truth, only then execute.
- **Bundled scripts** — script logic that is deterministic and would be
  regenerated identically on every run (validators, parsers, formatters).
  Correctness-critical logic deserves a tested script from the start;
  one-off or highly task-dependent logic stays inline.

## Progressive disclosure

- Design SKILL.md first; split `references/` by *branching condition*, not
  by topic. Content that is always needed together for one branch belongs in
  one file, even if it spans several sub-topics. Content needed on most runs
  belongs in the SKILL.md body.
- Every reference-load instruction states a precise trigger:
  `Read references/api-errors.md when the API returns a non-200 status.`
  Never "see references/ for details" — without a condition the agent either
  loads everything every run or nothing at all.

## Scripts

Rules for anything in a skill's `scripts/`:

- No interactive prompts — agents run in non-interactive shells and hang on
  TTY input. Accept all input via flags.
- Provide `--help`; it is how agents discover the interface.
- Data to stdout, diagnostics to stderr; structured output (JSON) where
  downstream steps consume it.
- Exit codes: 0 success, 1 failure, 2 bad arguments; document deviations in
  `--help`.
- Idempotent by default — agents retry; "create if absent" beats "fail on
  duplicate".
- Errors must say what is wrong, why it matters, and how to fix it.
- Introduce each script in SKILL.md with a relative markdown link at its
  first mention (`[`scripts/validate.py`](scripts/validate.py)`) so the
  agent knows the path is relative to the skill root, then show the
  invocation.
- Pin versions in one-off commands (`npx tool@1.2.3`) and state runtime
  prerequisites (or use the `compatibility` field) instead of assuming them.

## Checklist before committing

1. Scope: the skill covers one coherent work unit and adds something the
   agent lacks.
2. `description` states what + when, including indirect trigger phrasings.
3. No path escapes the skill directory; every skill it names is within the
   allowed dependency range and handed off through
   `ryan-minato-skills-installing`, with no install command (public skills).
4. The name was considered against the default shape and the catalog's
   `## Naming` section (recommended, not enforced).
5. SKILL.md holds the common path; branch-specific detail lives in
   `references/` behind precise load conditions; deterministic logic lives
   in `scripts/`.
6. Catalog README.md and README.zh.md both list the skill; the symlink in
   `.agents/skills/` exists (public skills).
7. When moving or renaming: every reference `git grep -n <old-name>` finds
   is updated in the same change.
8. `just check-skill <dir>` is clean, then `just check` passes.
