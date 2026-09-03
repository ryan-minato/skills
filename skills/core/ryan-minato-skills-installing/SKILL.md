---
name: ryan-minato-skills-installing
description: >
  Installs Agent Skills from the ryan-minato/skills library into a project or
  globally, and lists what the library offers. Use when installing or adding a skill
  from ryan-minato/skills — "install the git-commit skill", "add sensitivity-check
  globally"; when another skill hands off to this skill to install a skill or a
  whole catalog ("load the ryan-minato-skills-installing skill and install X as it
  directs"); when a skill or document instructs running "npx skills add
  ryan-minato/skills"; when asked what skills the library contains; or when an
  installed skill from this library needs updating to its latest upstream version. Not
  for npm packages, project dependencies, or any other skill library — nothing in it
  applies outside ryan-minato/skills.
license: Apache-2.0
compatibility: >
  Preferred path needs a Node package runner (pnpm, bun, yarn, or npx) with
  network access to fetch the skills CLI and the source repo. The bundled
  fallback (scripts/install_skill.py) needs git and Python 3.9+ (stdlib only,
  no uv/Node) plus network access to clone github.com/ryan-minato/skills.
---

# Installing skills from ryan-minato/skills

Install one or more skills from the `ryan-minato/skills` library
(<https://github.com/ryan-minato/skills.git>) reliably, whatever tooling the
machine has. Pick the highest-priority method that works, and never leave the
user with a half-installed or wrongly-scoped skill.

## Step 1 — Resolve name(s) and scope

Decide, before running anything:

- **Which skill(s)** — the exact directory name(s) the user wants (e.g.
  `git-commit`, `sensitivity-check`). If the user named a capability rather
  than a skill, or asked what exists, discover first (Step 2). When another
  skill handed off here, take the name(s) from its handoff sentence.
- **Whole catalog** — a handoff or request may name a catalog instead
  (`meta`, `scaffold`). `meta` is installed whole: its builders stack and are
  disposed together. `scaffold` is installed as one topic builder — the one
  whose topic matches the project — never every scaffold builder; the `meta`
  catalog's `meta-disposal` later removes both catalogs' builders. Both are
  project-scope, disposable components. Discover the members (Step 2); for
  `meta` install every member, for `scaffold` install only the chosen topic
  builder.
- **Scope** — **project** (default) installs into this project so the files
  commit with it; **global** installs into the user home dir for every
  project. Choose global when the user says "globally" / "for every project",
  or when installing a `core` skill (those are meant to load everywhere).
  Disposable catalogs (`meta`, `scaffold`) are always project scope, and
  unlike durable skills they are never committed: the builders keep their
  own directories out of every commit and `meta-disposal` removes them.

Done when: you have a concrete skill-name list and a project-or-global scope.

## Step 2 — Discover available skills (when the name is unknown)

List the library's inventory instead of guessing names:

- Node runner available → `<runner> skills add ryan-minato/skills --list`
  (see Step 3 for how to choose `<runner>`).
- Otherwise → the bundled script. Run
  [`scripts/install_skill.py`](scripts/install_skill.py) to list every skill
  with its catalog and description:

  ```bash
  python3 scripts/install_skill.py --list          # human-readable table
  python3 scripts/install_skill.py --list --json   # machine-readable
  ```

Done when: the target skill name(s) are confirmed against the real inventory.

## Step 3 — Install with the skills CLI (preferred)

Use the vercel-labs skills CLI through the first package runner present on the
machine — prefer modern runners, fall back to npx:

1. `pnpm dlx` — if `pnpm` is installed (preferred)
2. `bunx` — if `bun` is installed
3. `yarn dlx` — if `yarn` is installed
4. `npx` — the fallback when only npm/npx is present

Run the same subcommand with the chosen runner; `--skill` repeats, one per
skill:

```bash
# project scope — copy real files into ./.claude/skills so they commit with the project
pnpm dlx skills add ryan-minato/skills --skill <name> --copy -y

# global scope — install into the user home dir for every project
pnpm dlx skills add ryan-minato/skills --skill <name> -g -y

# whole catalog — one --skill per member from the Step 2 listing, project scope
pnpm dlx skills add ryan-minato/skills --skill <name-1> --skill <name-2> ... --copy -y
```

- **Project scope must pass `--copy`** so the skill lands as a real
  directory the project owns, like the repo's other project-level skills —
  not a symlink into a global cache. Durable skills then commit with the
  project; disposable builders do not.
- `-g` selects global; omit it for project.
- `-y` skips interactive prompts (agents hang on TTY input).
- The CLI knows skills, not catalogs: for a whole catalog, pass every member
  the listing shows under that catalog. Never use `--skill '*'` — it would
  install every catalog of the library.
- Swap `pnpm dlx` for `bunx` / `yarn dlx` / `npx` per the priority above.

If no Node runner exists at all, go to Step 4.

Done when: the runner exits 0 and the destination skill directory exists.

## Step 4 — Fallback when no Node runner is available

When the machine has no `pnpm`/`bun`/`yarn`/`npx`, install with the bundled
script, which clones the library to a temp dir and copies the skill folder to
the scope-appropriate location (real files, never a symlink):

```bash
python3 scripts/install_skill.py <name> [<name> ...]   # project scope (./.claude)
python3 scripts/install_skill.py <name> --global       # global scope (~/.claude)
python3 scripts/install_skill.py <name> --force        # replace if already present
python3 scripts/install_skill.py --catalog meta        # only meta is installed whole
python3 scripts/install_skill.py scaffold-ml          # scaffold: by name, the one topic builder
```

The script refuses `--catalog` for any other catalog and refuses `--global`
whenever a disposable builder is in the install set, however it was named
(exit 2 with the reason).

It needs `git` and Python 3.9+ only. Re-run with `--force` to update an
already-installed skill.

Done when: the script's JSON output reports each skill's copied destination.

## Verify

1. Confirm the destination exists as a **real file**, not a dangling symlink:
   - project → `./.claude/skills/<name>/SKILL.md`
   - global → `~/.claude/skills/<name>/SKILL.md`
2. Tell the user to reload their agent session so the new skill is
   discovered. Never report success without checking the path exists.

## Gotchas

- A project install must leave a real directory (`--copy` for the CLI; the
  fallback always copies) — a default CLI symlink points into a global cache,
  would not commit with the project, and cannot be disposed of by deleting it.
- `-g` writes to the user home agent dir, not the project.
- The fallback resolves a bare skill name by scanning all catalogs
  (`skills/*/<name>`); an unknown or ambiguous name is an error, so discover
  (Step 2) when unsure.
- Installed skills are copies; updating one means re-running the install
  (`--force` for the fallback).
- A non–Claude Code client uses a different agent dir — pass `--agent-dir`
  (fallback) or the CLI's `--agent` (e.g. `.codex`); default is `.claude`.
- Run the fallback script **from the project root**, not from inside the skill
  directory: a project-scope install copies into the current directory's agent
  dir (`./.claude`), while the script's own file lives inside the installed
  skill (e.g. `.claude/skills/ryan-minato-skills-installing/scripts/install_skill.py`).
  `cd`-ing into that directory first would land a project install in the wrong
  place.
