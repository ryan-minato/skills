## Why

`scripts/lint_skill.py` in `great-skill-writing` imports PyYAML at module
level, so on a host without it (no `uv`, or the script started with plain
`python3`) `--help`, an unknown option, and a missing skill path all die
with a `ModuleNotFoundError` traceback and exit 1 — the exact
`python3 scripts/lint_skill.py --help` smoke test that every other script
in the library passes. The script rules in
`.agents/knowledge/skill-quality.md` make `--help` the way agents discover
a script's interface, so a `--help` that needs a dependency to answer is
wrong installed behavior. Now: it is the first behavior change to a public
skill after the harness rebuild, so it is also the OpenSpec pilot (#76).

## What Changes

- `great-skill-writing`: `scripts/lint_skill.py` imports PyYAML only when
  it parses a SKILL.md, so `--help`, unknown options, and a missing skill
  path need nothing beyond the standard library; a real lint on a host
  without PyYAML exits 2 with a diagnostic naming the dependency and how to
  get it, instead of a traceback; `--help` documents that exit code. Not
  BREAKING: the documented `uv run` path behaves exactly as before.

## Skills touched

- `core/great-skill-writing` (new): the domain's description triggers and
  the contract of `lint_skill.py`.

## Installed behavior

An agent that runs the linter on a host without PyYAML now gets usage,
argument diagnostics, and an actionable dependency message instead of a
traceback; every other path is unchanged → `fix`.

## Impact

None of the synchronized surfaces move: the description is unchanged, so
the `core` README pair rows stay; no file is added, moved, or removed, so
the `.agents/skills/` symlink, `marketplace.json`, `skills/core/CONTEXT.md`,
and the mirrors in `scripts/validate_harness.py` are untouched. The change
creates the domain `openspec/specs/core/great-skill-writing/` at archive.

## Non-goals

- Changing SKILL.md, its `compatibility: Validation requires uv.` line, or
  `references/scripts-python.md`; the deferred-import pattern as authoring
  guidance is a separate capability change.
- Specifying the rest of `great-skill-writing` (its `Behavior:` and
  `Handoff:` requirements): specs are never backfilled.
- Items 2–4 of #77 (dev container extension, MCP declaration, merged
  branches); they are the repository change `post-rebuild-cleanup`.

## Tracked work

#76 (the OpenSpec pilot; #78 was the first end-to-end run of the loop, this
change is the first run of the lifecycle as it now stands) and item 1 of
#77.
