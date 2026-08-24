# Repository guardrail operations

Read when applying or inspecting protected branches or tags, merge-request
approval rules, security settings, or other GitLab project guardrails. Author
local CODEOWNERS, Renovate, pipeline, and scanning configuration through
`gitlab-community` first.

## Safety sequence

1. Resolve the host, full project path, instance version, license tier,
   visibility, default branch, current protections and approvals, available
   scanning features, and the authenticated actor's role.
2. Fetch current official documentation for the target instance version and
   selected endpoint. GitLab capabilities and request shapes vary by version
   and tier.
3. Present an exact before/after plan including protected patterns, allowed
   roles or users, approval counts and resets, CODEOWNERS behavior, scanning
   implications, and rollback.
4. Run the normal pre-publish gate over the reviewed payload.
5. Apply only after explicit approval, using the matching MCP capability when
   available or `glab api` with file-backed payloads.
6. Read the settings back and compare them with the plan. A 404 may indicate a
   tier-gated feature; report the requirement instead of retrying broader
   permissions or alternate destructive endpoints.

## Boundaries

- Protection and approval features may be group-inherited. Do not silently
  shadow a group policy at project level.
- CODEOWNERS enforcement and required approval rules are tier-dependent.
- Scanning commonly becomes active through CI templates; remote settings do
  not replace the local pipeline authored by `gitlab-community`.
- Removing protection, lowering approval counts, or disabling scanning is a
  destructive policy change and always needs fresh confirmation.

Done when: read-back matches the approved plan, rollback is recorded, and
every tier-, role-, or group-blocked action is reported.
