# Repository guardrail operations

Read when applying or inspecting GitHub rulesets, branch/tag protections,
security settings, or other repository-level guardrails. Author the desired
policy and local files through `github-community` first.

## Safety sequence

1. Read current repository visibility, default branch, rulesets, required
   checks, installed apps, security feature status, and the authenticated
   actor's administrative permission.
2. Fetch the current official GitHub documentation for the selected endpoint
   and feature. Availability and request shapes change; do not recall them from
   memory.
3. Present an exact before/after plan. Name enforcement mode, targets, bypass
   actors, required checks and reviews, feature plan requirements, and rollback.
4. Run the normal pre-publish gate over the reviewed JSON or settings payload.
5. Apply only after explicit user approval. Use the matching MCP capability
   when present; otherwise use `gh api` with a payload file.
6. Read back the resulting settings and compare them field by field with the
   reviewed plan. Report unsupported, tier-gated, or permission-blocked fields
   without retrying a broader change.

## Boundaries

- Repository rulesets and legacy branch protection overlap. Prefer one
  reviewed ownership model; do not install conflicting enforcement.
- A required status check must already be produced on the target branch.
- CODEOWNERS and Dependabot files are local authoring owned by
  `github-community`; this branch only verifies their platform effects.
- Enabling security features may expose findings or consume paid capacity.
  Confirm visibility, licensing, and notification ownership before changing
  them.
- Removing or weakening a protection is destructive policy change and always
  requires fresh confirmation.

Done when: read-back matches the approved plan, rollback is recorded, and every
remaining manual or plan-gated action is reported.
