## Why

The `great-skill-writing` description promises to load "when a skill misbehaves — never triggers, fires on the wrong task, or produces inconsistent output", but at the Sonnet tier it did not load for exactly that request. Given "the instruction package I gave my agent for changelog entries never gets picked up — fix it", or the direct "my changelog-entries skill never triggers …", the solver treated the problem as a discovery path, added a symlink, and loaded nothing (#82). The `lint-skill-help` change dropped that scenario from the domain for this reason and left the weakness to this issue.

## What Changes

- `great-skill-writing`: the description is rewritten so that a request about a skill or agent instruction package that never gets picked up or never triggers loads the skill, in both direct and indirect phrasing. Requests the description already covers still load it, and its exclusions still hold. A request about something that is not a skill and never triggers, such as a git hook, does not load it. Only the frontmatter `description` changes. Not BREAKING.

## Skills touched

- `core/great-skill-writing` (modified): the `Trigger: description` requirement. It restores the indirect-phrasing scenario removed in cb49b76 and adds a direct-phrasing scenario and a non-skill near-miss.

## Installed behavior

An agent asked to fix a skill that never loads now loads the authoring skill and can diagnose the skill's description. Before, it answered from priors and at most moved or relinked the skill. The description already claimed this case, so the change corrects wrongly restrictive behavior → `fix`.

## Impact

None of the synchronized surfaces move. The `core` README pair rows (`skills/core/README.md`, `README.zh.md`) describe the skill's capability, not its trigger wording, and stay unchanged. No file is added, moved, or removed, so the `.agents/skills/` symlink, `marketplace.json`, `skills/core/CONTEXT.md`, and the mirrors in `scripts/validate_harness.py` are untouched. At archive the modified block replaces `Trigger: description` in `openspec/specs/core/great-skill-writing/spec.md`.

## Non-goals

- The SKILL.md body and its references. In particular, `references/failure-modes.md` says "trigger failures are always description failures" and never mentions the directories a client scans for skills. That wording is a proposed follow-up, not part of this change.
- `Script: lint_skill.py` and the bundled linter.
- The descriptions of neighbouring skills (`meta-harness`, `skill-authoring`).
- The globally installed `great-skill-writer` copy that carries the old description. It lives outside the repository, and the maintainer updates or removes it (see design.md, Verification plan).

## Tracked work

#82, found while running `lint-skill-help` (#80) as the OpenSpec pilot (#76).
