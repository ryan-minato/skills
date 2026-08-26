# Versioning Policy

<!-- Rework every slot. The tag-check workflow READS the Tag pattern line
below — keep the exact "Tag pattern: `...`" format or the check fails
loudly. -->

- Scheme: {{SCHEME — e.g. SemVer 2.0.0}}; version source:
  {{VERSION_SOURCE — the one file that owns the number}}.
- Tag format: {{TAG_FORMAT — e.g. vMAJOR.MINOR.PATCH}}.

Tag pattern: `{{TAG_REGEX — e.g. ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$}}`

- Bump rules: {{BUMP_RULES — what forces major, minor, patch}}.
- Release gate: {{RELEASE_GATE — what must be green before tagging}}.
- Procedure: {{RELEASE_PROCEDURE — draft-first when immutable releases
  are on; generated notes from PR labels via .github/release.yml}}.
- Rollback: {{ROLLBACK — how a bad release is superseded; tags are never
  deleted}}.
