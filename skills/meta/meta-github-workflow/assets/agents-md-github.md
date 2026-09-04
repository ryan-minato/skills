<!-- The `## GitHub lifecycle` section to splice into the target's
AGENTS.md (or its existing entrypoint). Rework every slot; keep it a map,
not a manual. -->

## GitHub lifecycle

- Repository: `{{OWNER_REPO}}` on {{HOST}} · enforcement tier:
  {{ENFORCED_ADVISORY_OR_CONVENTION}}.
- Recurring issue, pull-request, release, and platform work → the
  `github-project-workflow` skill in {{SKILL_DIR_PATH}}.
- Branches, issues, pull requests, label meanings, tracking-issue
  conventions, and milestone policy → `.agents/knowledge/github-workflow.md`
  before creating a branch, opening or editing an issue or pull request, or
  creating a planning object.
- CI job ↔ command map and what a healthy run looks like →
  {{KNOWLEDGE_PATH}}/checks.md before touching workflows or diagnosing a
  red check. Never weaken or delete a check to make it pass.
- Remote settings (rulesets, protection, environments) →
  {{KNOWLEDGE_PATH}}/platform-settings.md; changing any is a human-approved
  action.
- Publish only with `SAFE TO PUBLISH: YES` on the exact payload and
  applicable external-write approval. Published content cannot be
  reliably erased.
- Update triggers: {{SYNC_TRIGGER_SUMMARY — e.g. "template or taxonomy
  edits update checks.md and labels.json in the same PR"}}.
