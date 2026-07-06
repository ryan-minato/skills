# devcontainer-template.json field reference

Read this when you need a field not covered in SKILL.md or are reviewing
an existing template manifest. Authoritative source:
<https://raw.githubusercontent.com/devcontainers/spec/main/docs/specs/devcontainer-templates.md>

## Identity (required)

| Field | Rules |
|---|---|
| `id` | Must equal the `src/<id>/` directory name; last segment of the OCI ref. |
| `version` | Semver; publishing only happens when it changes (tags `:MAJOR`, `:MAJOR.MINOR`, `:MAJOR.MINOR.PATCH`, `:latest`). |
| `name` | Human-readable display name. |

## Descriptive

| Field | Notes |
|---|---|
| `description` | Shown in pickers and the generated README. |
| `documentationURL` | Where "learn more" points. |
| `licenseURL` | License link. |
| `publisher` | Display name of the publishing person/org. |
| `keywords` | Array of search terms. |
| `platforms` | Languages/stacks the template targets (e.g. `["Python", "Node.js"]`, or `["Any"]`) — declare honestly; pickers filter on it. |

## options

Same grammar as Features: each key maps to
`{ "type": "string"|"boolean", "default", "description", "proposals": [...] | "enum": [...] }`.

- Every option needs a working `default` — tools may apply with zero
  input, and smoke tests substitute defaults.
- `${templateOption:<key>}` placeholders in any text file of the payload
  are replaced with the chosen value at apply time. Plain text
  substitution: no conditionals, no loops, no escaping mechanism, and no
  substitution inside binary files.
- An option that only ever feeds an image tag is often better expressed
  as `imageVariant` with `proposals`, letting users type arbitrary tags.

## optionalPaths

Array of paths (files, directories, or globs relative to the template
root, e.g. `[".github/*", "docs"]`) that applying tools may offer to
exclude. Use for genuinely removable extras — sample workflows, example
code — never for files the container needs to start.

## Files and payload

Everything under `src/<id>/` except `devcontainer-template.json`,
`README.md`, and `NOTES.md` is the payload copied into the user's
project — typically `.devcontainer/devcontainer.json` (or
`.devcontainer.json`) plus any Dockerfile/compose files it references.
The manifest itself is not copied.
