## Context

See proposal.md. Constraints: `scripts/check_pr_policy.py` reads the
template's `## ` headings from the base branch, so the new body shape is
enforced from the pull request after this one; `scripts/validate_harness.py`
requires the template to carry headings for related, validation, and
checklist, an unticked `secrets` line, and a `Spec:` line, and fails when
`ARCHITECTURE.md` names a path that does not exist (so the MCP file and its
row go together); the ruleset write of 2026-09-10 landed everything except
the bypass actor, which the REST API refuses on a user-owned repository, so
the in-request archive rule stays in force and this change archives itself
before ready; the `openspec-*` skills are generated and untouched.

## Placement

| What Changes bullet | File(s) | Check that proves it |
|---|---|---|
| Dev container | `.devcontainer/devcontainer.json` extensions list | `grep -c deno .devcontainer/devcontainer.json` prints 0; `just validate` (ruff pin still read) |
| MCP declaration | delete `.agents/mcp_config.json`; `ARCHITECTURE.md` path table; `.agents/knowledge/harness-maintenance.md` MCP row | `just validate` (pointers); `grep -rn agentskills.io --include='*.json' --include='*.toml' .` lists exactly three declarations, matching the row |
| Settings register | `.agents/knowledge/github-settings.md` bypass row and Last verification | readback commands in the row reproduce the recorded state; the API refusal is quoted |
| Specification boundary wording | `.agents/knowledge/spec-workflow.md` artifact table (rows for the schema and the change directory); `ARCHITECTURE.md` Specifications; `openspec/schemas/skill-change/schema.yaml` description lines and the proposal and specs instructions; `templates/proposal.md`, `templates/design.md`; `AGENTS.md` Always; `.github/ISSUE_TEMPLATE/bug-report.yml` description | `git grep -n -i 'skill or tool\|skills or tools\|specs/repository\|Tool:' -- AGENTS.md ARCHITECTURE.md .agents/knowledge openspec/schemas .github/ISSUE_TEMPLATE` prints nothing; `just spec-validate` (schema still parses the archived and in-flight changes); `just validate` |
| Pull request body shape | `.github/PULL_REQUEST_TEMPLATE.md`; `scripts/check_pr_policy.py` (headings from the template as before; `Phase: implementation` and no reserved placeholder in Changes or Validation on a ready pull request; docstring); `scripts/validate_harness.py` template roles; `.agents/skills/change-workflow/SKILL.md` §5–§7 (payload built from the template, reserved sections filled at ready, permalinks, extra sections through the same gate); `.agents/knowledge/spec-workflow.md` Specifications and tracked work; `.agents/knowledge/agent-authority.md` report; `harness-maintenance.md` new row | `python3 scripts/check_pr_policy.py --pr <n>` dry runs against fixture bodies (see the plan); `just validate`; a clean-context readback of `change-workflow` §6 reproduces the shape |
| Approval record | `.agents/knowledge/spec-workflow.md` Lifecycle step 2; `change-workflow` §3; the template's Specification comment and checklist item | readback: the three places say `Specification approved` and the timeline rule; `git grep -n 'approved at'` prints nothing under `.agents/`, `.github/` |
| Remote branches | `origin` (a remote operation, no file) | `git branch -r --merged origin/main \| grep -v main` prints nothing; `git ls-remote --heads origin` lists only `main` and the open pull request branches |

## Decisions

- **One repository change, not several**: every item is a leftover or a
  register readback with no design choice inside it; splitting would cost
  more approvals than it saves review. Alternative rejected: a separate
  `pr-body-shape` change.
- **Template enforcement stays heading-based**: `check_pr_policy.py` keeps
  reading the template's headings, so the template remains the single
  source; the script adds only the ready-state rules the shape implies
  (implementation phase, reserved sections filled). Alternative rejected: a
  fixed heading list in the script, which would be a second source.
- **Reserved sections carry a visible placeholder** (`_Reserved: …_`)
  rather than an HTML comment, so a reader sees what will arrive; the
  script recognizes the placeholder and rejects a ready pull request that
  still holds it. Alternative rejected: empty sections, which a reader
  cannot distinguish from an omission.
- **The approval comment names no commit**; the timeline is the record. A
  push to the record after the comment needs a fresh comment; the ready
  conditions say so. Alternative rejected: keeping `at <sha>`, which the
  maintainer found error-prone to type and redundant with the timeline.
- **The bypass stays pending, honestly recorded**: the register quotes the
  API refusal and names the two paths (ruleset UI, deploy key) without
  choosing; choosing is a maintainer decision outside this change.
- **This change archives in-request** because the bypass is not granted;
  the `Spec:` link is refreshed to the archive path at ready.

## Risks / Trade-offs

- [The template change lands while #80 is open on the old headings] → #80
  edits its body to the new template after this merges; its policy check
  runs on the base branch's template, so it turns green on that edit.
- [The policy script gains a placeholder rule that a future body could
  trip by quoting it] → the rule matches only a line that is entirely the
  placeholder.
- [Deleting `feat/harden-github-harness` loses three unmerged commits] →
  they remain reachable at `refs/pull/69/head`; the pull request body says
  so before the deletion.
- [Schema text edits] → strict validation runs over every archived and
  in-flight change after the edit.

## Verification plan

Per What Changes bullet, the command or observation that proves it:

- Dev container: `grep -c deno .devcontainer/devcontainer.json` → `0`;
  `python3 -c "import json;json.load(open('.devcontainer/devcontainer.json'))"`
  exits 0; `just validate` passes.
- MCP declaration: `test ! -e .agents/mcp_config.json`;
  `grep -rln agentskills.io --include='*.json' --include='*.toml' .` lists
  `.mcp.json`, `.codex/config.toml`, `.devcontainer/devcontainer.json` and
  nothing else; `just validate` passes with the `ARCHITECTURE.md` row gone.
- Settings register: `gh api repos/ryan-minato/skills/rulesets/19602018`
  readback matches the ruleset row and the Last verification paragraph
  (contexts, strict, thread resolution, merge methods, unattributed, approvals,
  `bypass_actors: []`); `gh api repos/ryan-minato/skills/branches/main/protection`
  → 404.
- Specification boundary wording: the `git grep` in the placement table
  prints nothing; `just spec-validate` passes; a clean-context subagent
  given `AGENTS.md`, `ARCHITECTURE.md` Specifications, and the schema's
  proposal instruction answers "where does a change to `scripts/` get its
  spec?" with "nowhere — it is a `skip_specs` repository change".
- Pull request body shape: `python3 scripts/check_pr_policy.py` on fixture
  event payloads built from the new template — (a) draft with reserved
  sections → 0 findings; (b) ready with reserved sections still present →
  findings name Changes and Validation; (c) ready with `Phase:
  specification` → finding names the phase; (d) ready with both sections
  filled, `Phase: implementation`, every box ticked → 0 findings; (e) a
  body missing `## Why` → finding names it. `just validate` passes with the
  new template. A clean-context readback of `change-workflow` §6 lists the
  sections in order and states the permalink and reserved-section rules.
- Approval record: `git grep -n 'approved at' -- .agents .github` prints
  nothing; the readback of `spec-workflow.md` Lifecycle step 2 states the
  comment text and the timeline rule.
- Remote branches: `git branch -r --merged origin/main | grep -v main`
  prints nothing; `git ls-remote --heads origin` shows no deleted branch.
- `just check` passes.

Skipped: none.
