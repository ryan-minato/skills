---
name: design-md
description: >-
  Disposable meta-skill (delete after the harness is built): authors or
  edits DESIGN.md, the strict visual-design description format — YAML
  front-matter design tokens plus a prose body — that makes a project's
  visual language readable to agents, with a bundled OKLCH color
  calculator. Use when a project's visual design must be encoded for
  agents, or a DESIGN.md needs creating, linting, or updating. Not for
  any other document — the DESIGN.md name is reserved for this format
  and never repurposed.
---

# DESIGN.md

This skill produces the project's `DESIGN.md` following the public
DESIGN.md format: optional YAML front matter carrying the normative design
tokens, and a prose body — the prose is where the design lives, the tokens
only pin its values. The file serves agents (and designers) across
sessions; the name is reserved for this format and must never be used for
anything else.

## Workflow

1. Gather the design's sources of truth: existing CSS variables, token
   files, brand guides, current screens. Derive the design from them —
   invent nothing the project cannot confirm.
2. When creating the file or changing its front-matter or section
   structure, read [format-spec.md](references/format-spec.md) first, and
   start new files from
   [design-md-skeleton.md](assets/design-md-skeleton.md): copy it, rework
   every section against the real design, delete sections the project
   does not need, and remove all placeholder text.
3. Keep the core rules while writing: front matter needs `name` and at
   least a `primary` color when tokens are present; `{dot.path}`
   references resolve to primitives (composites only inside
   `components`); body sections keep the spec order — Overview, Colors,
   Typography, Layout, Elevation & Depth, Shapes, Components, Do's and
   Don'ts — omitting freely but never duplicating a heading; hex is the
   recommended default color format.
4. Validate colors with `scripts/oklch.py` whenever the design uses
   `oklch()` or other wide-gamut notation, or the prose commits to
   contrast ratios: `to-hex` / `from-hex` convert, `gamut` flags values
   outside sRGB (the linter converts colors to sRGB for WCAG checks, so
   out-of-gamut OKLCH silently clips), `contrast` reports WCAG ratios.
   The calculator is a compatible extra, not something the format
   requires.
5. Lint: `npx @google/design.md lint DESIGN.md` when a Node runtime is
   available; fix every error. Without Node, verify manually against the
   checklist in [format-spec.md](references/format-spec.md).
6. Register `DESIGN.md` in the entrypoint's when-to-read table, and add
   its keep-current rule to the harness's sync mechanism: a change to the
   design tokens or visual language updates `DESIGN.md` in the same
   change.

Done when: `DESIGN.md` lints clean (or passes the manual checklist), every
committed contrast pair passes, and the entrypoint points at the file with
a load condition.

## Gotchas

- Never repurpose the `DESIGN.md` name for architecture notes, decision
  records, or anything else — agents and tools assume the format on
  sight.
- A token dump with empty prose fails the format's intent: tokens are
  normative, but the prose carries the design.
- Duplicate section headings are a hard error, even under the tolerated
  synonym titles.
- An `oklch()` value outside sRGB passes unnoticed until the linter's
  sRGB conversion clips it — run `gamut` before committing wide-gamut
  tokens.
- A `{dot.path}` reference to a missing or non-primitive token breaks the
  normative layer; re-check references after any token rename.
