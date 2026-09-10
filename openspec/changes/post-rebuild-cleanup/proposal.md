## Why

The harness rebuild (#74) left four items untouched on purpose, and two
things happened since: the maintainer authorized the ruleset write that
#75 planned, whose readback the settings register must now carry, and the
first OpenSpec pilot found five places where the harness still says
repository tools have spec domains, although its authoritative rules
(`openspec/config.yaml`, the Domains section of `spec-workflow.md`,
`ARCHITECTURE.md`) give a domain to public skills only. This repository
change closes the leftovers, records the remote state, and makes the
specification boundary read the same everywhere.

## What Changes

- Dev container: the `denoland.vscode-deno` extension goes; the repository
  has no Deno code, configuration, or runtime.
- MCP declaration: `.agents/mcp_config.json` goes — no client convention
  reads it; the endpoint stays declared once per client that does
  (`.mcp.json`, `.codex/config.toml`, `.devcontainer/devcontainer.json`),
  and the register row in `harness-maintenance.md` and the path table in
  `ARCHITECTURE.md` say so.
- Settings register: `github-settings.md` records the 2026-09-10 readback
  of ruleset `Default` (required checks `checks / gate`, `pr / policy`,
  `scan-secrets`, strict; review threads must be resolved; squash and
  rebase only; extra approval for unattributed changes off) and the fact
  that the REST API refuses the GitHub Actions app as a bypass actor on a
  user-owned repository, so the in-request archive rule stays in force
  until another path grants the push.
- Specification boundary wording: `spec-workflow.md` drops the
  `specs/repository/<tool>/` path and the `Tool:` requirement kind that the
  schema never defined (`ARCHITECTURE.md` likewise); the `skill-change`
  schema and its templates say "skill" where they said "skill or tool" and
  point repository tooling at the repository change; `AGENTS.md` states the
  boundary in its Always list; the bug report form separates a skill's spec
  from a tool's documented intent.
- Remote branches: `feat/project-code-review` (merged, #68) and
  `feat/harden-github-harness` (closed unmerged, #69; its commits remain
  reachable at `refs/pull/69/head`) are deleted from `origin`, so no merged
  branch remains there.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository read one statement of the specification
boundary — a public skill has a spec domain, everything else is a
`skip_specs` repository change — see the live ruleset state in the
register, and stop looking for a fourth MCP declaration or a Deno toolchain.

## Impact

- Deleted: `.agents/mcp_config.json`.
- Edited: `.devcontainer/devcontainer.json`, `ARCHITECTURE.md`, `AGENTS.md`,
  `.agents/knowledge/github-settings.md`, `.agents/knowledge/spec-workflow.md`,
  `.agents/knowledge/harness-maintenance.md`,
  `openspec/schemas/skill-change/schema.yaml` and
  `templates/proposal.md`, `templates/design.md`,
  `.github/ISSUE_TEMPLATE/bug-report.yml`.
- Remote: two branches deleted on `origin` (listed above).
- `scripts/validate_harness.py` checks that every path `ARCHITECTURE.md`
  names exists, so the MCP row and the deletion land together.

## Non-goals

- Changing how the MCP server is used, or the `backup-*` tags.
- Required approvals (stays 0), CODEOWNERS, releases.
- Granting the archive workflow a push path (UI-side bypass or a deploy
  key): a maintainer decision proposed separately.
- `lint_skill.py` (#77 item 1): the skill change `lint-skill-help`.

## Tracked work

#75 (ruleset readback) and #77 (items 2–4).
