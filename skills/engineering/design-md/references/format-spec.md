# DESIGN.md Format — Condensed Spec

Read when creating a DESIGN.md, changing its front matter or section
structure, or verifying manually because no Node runtime is available.
Upstream (normative, version "alpha"):
[google-labs-code/design.md](https://github.com/google-labs-code/design.md).
This file condenses it and must be updated when the upstream spec changes.

## Shape

One file, two parts: optional YAML front matter (machine-readable design
tokens; the normative values) and a markdown body (design rationale and
application guidance). Prose may use descriptive color names as long as they
correspond to token names.

## Front-Matter Schema

```yaml
version: <string>          # optional; current: "alpha"
name: <string>             # required when front matter is present
description: <string>      # optional
omitted: <string[] | OmittedSection[]> # optional
colors:
  <token-name>: <Color>    # at least primary when colors are defined
typography:
  <token-name>: <Typography>
rounded:
  <scale-level>: <Dimension>
spacing:
  <scale-level>: <Dimension | number>
components:
  <component-name>:
    <property>: <string | token reference>
```

- **Color** — any valid CSS color: hex, named, `rgb()`/`hsl()`/`hwb()`,
  wide-gamut `oklch()`/`oklab()`/`lch()`/`lab()`, `color-mix(in srgb, …)`.
  Hex `#RRGGBB` is the recommended default. All colors are converted to
  sRGB internally for WCAG contrast checking; the original string is
  preserved for display and export.
- **Dimension** — a string with unit `px`, `em`, or `rem`.
- **Typography** — `fontFamily` (string), `fontSize` (Dimension),
  `fontWeight` (number or quoted numeric string), `lineHeight` (Dimension or
  unitless multiplier),
  optional `letterSpacing` (Dimension), `fontFeature`, `fontVariation`
  (strings).
- **Token references** — `{path.to.token}`; must point to a primitive
  (`{colors.primary-60}`), not a group. Inside `components`, references
  to composites like `{typography.label-md}` are permitted.
- **Component properties** — `backgroundColor`, `textColor` (Color),
  `typography` (Typography), `rounded`, `padding`, `size`, `height`,
  `width` (Dimension). Variants are sibling keys: `button-primary`,
  `button-primary-hover`, `button-primary-active`.

## Body Sections

All sections are `##` headings; an optional `#` title is not parsed as a
section. Omit freely, but those present keep this order, and a duplicated
heading rejects the file:

1. Overview (synonym: "Brand & Style") — personality, audience, feel
2. Colors — palettes; when present, define at least primary; common roles:
   primary, secondary, tertiary, neutral
3. Typography — levels (commonly 9–15); semantic names like headline,
   body, label
4. Layout (synonym: "Layout & Spacing") — grid or spacing strategy
5. Elevation & Depth (synonym: "Elevation") — or the flat-design
   alternative for hierarchy
6. Shapes — corner language; pairs with the `rounded` tokens
7. Components — per-component guidance; spec still evolving, extra
   components welcome
8. Do's and Don'ts — concrete guardrails

Unknown section headings and unknown token names are preserved or
accepted, not errors (unknown component properties warn).

## Tooling

`npx @google/design.md`:

- `lint <file>` — structure and token validation
- `diff <a> <b>` — token-level comparison
- `export <file>` — Tailwind or W3C design-token output
- `spec` — prints the full upstream spec

On Windows or PowerShell, use `npx -p @google/design.md designmd lint
DESIGN.md`; the dot-free alias avoids the `.md` command-resolution conflict.

## Manual Checklist (No Node)

1. If present, front matter opens and closes with bare `---`, parses as YAML,
   includes `name`, and uses `colors.primary` whenever colors are defined.
   Record intentionally absent token groups in `omitted` when applicable.
2. Every `{dot.path}` reference resolves; primitives only, except
   composites inside `components`.
3. Dimensions carry `px`/`em`/`rem`; `fontWeight` is numeric or a quoted
   numeric string.
4. Section headings are `##`, in spec order, no duplicates.
5. Prose exists for every section present and matches the tokens it
   describes.
6. Contrast pairs the prose commits to actually pass (use the bundled
   calculator).
