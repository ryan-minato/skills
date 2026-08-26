# DESIGN.md Format — Condensed Spec

Read when creating a DESIGN.md, changing its front matter or section
structure, or verifying manually because no Node runtime is available.
Upstream (normative, version "alpha"):
[google-labs-code/design.md](https://github.com/google-labs-code/design.md)
and its [format specification](https://github.com/google-labs-code/design.md/blob/main/docs/spec.md).
CLI behavior in this reference follows the current
[CLI implementation](https://github.com/google-labs-code/design.md/tree/main/packages/cli/src/commands).
Update this file when upstream changes.

## Shape

One file, two parts: optional YAML front matter (machine-readable design
tokens; the normative values) and a markdown body (design rationale and
application guidance). Prose may use descriptive color names as long as they
correspond to token names. A prose-only file is valid, but the current CLI
reports `No YAML content found` as a warning.

## Front-Matter Schema

```yaml
version: <string>          # optional; current: "alpha"
name: <string>             # optional
description: <string>      # optional
omitted: <string[] | OmittedSection[]> # optional
colors:
  <token-name>: <Color>    # include primary when colors are defined
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
- **Omitted** — an array of `colors`, `typography`, `spacing`, `rounded`, or
  `components`; each item may instead be `{ section: <name>, reason: <text> }`.
  Do not list a group that defines tokens. `spacing` and `rounded` omissions
  suppress their respective missing-group notices.

## Body Sections

All sections are `##` headings; an optional `#` title is not parsed as a
section. Omit freely, but those present keep this order, and a duplicated
heading rejects the file:

1. Overview (synonym: "Brand & Style") — personality, audience, feel
2. Colors — palettes; when present, include primary to avoid a warning and
   keep control of agent-generated key colors; common roles: primary,
   secondary, tertiary, neutral
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

`npx @google/design.md` (JSON output by default):

- `lint <file>` — structure and token validation; exits 1 only for errors,
  so review warnings even when it exits 0.
- `diff <before> <after>` — token-level comparison; exits 1 when the second
  file adds errors or warnings.
- `export --format <format> <file>` — `css-tailwind` (Tailwind v4),
  `json-tailwind` or `tailwind` (Tailwind v3), `dtcg`, or `css-vars` (with
  optional `--prefix`). Export succeeds even if the source has lint findings;
  lint separately.
- `spec` — prints the full upstream spec

On Windows or PowerShell, use `npx -p @google/design.md designmd lint
DESIGN.md`; the dot-free alias avoids the `.md` command-resolution conflict.

## Manual Checklist (No Node)

1. Front matter is optional. If present, it opens and closes with bare `---`,
   parses as YAML, and uses `colors.primary` whenever colors are defined.
   `name` is optional. Record intentionally absent token groups in `omitted`
   with only the supported group names and, when useful, a reason.
2. Every `{dot.path}` reference resolves; primitives only, except
   composites inside `components`.
3. Dimensions carry `px`/`em`/`rem`; `fontWeight` is numeric or a quoted
   numeric string.
4. Section headings are `##`, in spec order, no duplicates.
5. Prose exists for every section present and matches the tokens it
   describes.
6. Contrast pairs the prose commits to actually pass. Use the bundled
   calculator only for opaque hex and OKLCH pairs; otherwise use the linter.
