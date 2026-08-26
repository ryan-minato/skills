---
name: design-md
description: >-
  Authors and validates DESIGN.md, a durable visual-design specification
  combining optional YAML design tokens with prose guidance, with a bundled
  OKLCH color calculator. Use when a project's visual design must be encoded
  for agents, or a DESIGN.md needs creating, linting, updating, or exporting
  to token artifacts. Not for any other document — the DESIGN.md name is
  reserved for this format and never repurposed.
---

# DESIGN.md

This skill produces the project's `DESIGN.md` following the public
DESIGN.md format: optional YAML front matter carries normative design tokens,
and the prose body provides rationale and application guidance. The file
serves agents (and designers) across sessions; the name is reserved for this
format and must never be used for anything else.

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
3. Keep the core rules while writing: front matter is optional; `name` and
   `description` are optional metadata. When defining colors, include
   `primary` to avoid the linter's warning and keep control of the palette.
   Use `omitted` only for intentionally absent token groups. `{dot.path}`
   references resolve to primitives (composites only inside `components`);
   body sections keep the spec order — Overview, Colors, Typography, Layout,
   Elevation & Depth, Shapes, Components, Do's and Don'ts — omitting freely
   but never duplicating a heading; hex is the recommended default color
   format.
4. Validate colors with `scripts/oklch.py` whenever the design uses
   `oklch()` or the prose commits to contrast ratios expressed as opaque hex
   or OKLCH: `to-hex` / `from-hex` convert, `gamut` flags values outside sRGB
   (the linter converts colors to sRGB for WCAG checks, so out-of-gamut OKLCH
   silently clips), and `contrast` reports WCAG ratios. The calculator does
   not accept every CSS color or translucent color; use the upstream linter
   for those. It is a compatible extra, not something the format requires.
5. Validate with the upstream CLI when Node is available:
   - Run `npx @google/design.md lint DESIGN.md` and inspect its JSON summary.
     Fix every error; resolve each warning or record why it is intentional.
     Exit code 0 only means there are no errors, not that there are no
     warnings. A prose-only DESIGN.md is allowed but currently emits the
     `No YAML content found` warning.
   - Before replacing an existing DESIGN.md, save the revision as a candidate
     and run `npx @google/design.md diff DESIGN.md DESIGN.next.md`. The
     command exits 1 when the candidate adds errors or warnings; resolve or
     explicitly accept every regression before replacing the file.
   - When the project explicitly needs tokens in another format, run
     `export` with `css-tailwind`, `json-tailwind`, `dtcg`, or `css-vars`
     (the latter accepts `--prefix`). Export success does not validate its
     source; lint first.
   Without Node, verify manually against the checklist in
   [format-spec.md](references/format-spec.md).
6. Register `DESIGN.md` in the entrypoint's when-to-read table, and add
   its keep-current rule to the harness's sync mechanism: a change to the
   design tokens or visual language updates `DESIGN.md` in the same
   change.

Done when: `DESIGN.md` has no linter errors (or passes the manual checklist),
every warning and `diff` regression is resolved or recorded as intentional,
every committed contrast pair the bundled calculator supports passes, and the
entrypoint points at the file with a load condition.

## Gotchas

- Never repurpose the `DESIGN.md` name for architecture notes, decision
  records, or anything else — agents and tools assume the format on
  sight.
- A token dump with empty prose fails the format's intent: tokens are
  normative, but the prose carries the design.
- Lint warning-only output exits 0. Read the JSON summary instead of treating
  a successful command as an unqualified pass.
- Duplicate section headings are a hard error, even under the tolerated
  synonym titles.
- An `oklch()` value outside sRGB passes unnoticed until the linter's
  sRGB conversion clips it — run `gamut` before committing wide-gamut
  tokens.
- A `{dot.path}` reference to a missing or non-primitive token breaks the
  normative layer; re-check references after any token rename.
- `export` can produce an artifact from a source with lint findings; never use
  its exit code as the quality gate for DESIGN.md.
