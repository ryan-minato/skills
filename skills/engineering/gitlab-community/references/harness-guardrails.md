# GitLab harness guardrails

Read when authoring local CODEOWNERS, Renovate, scanning jobs, or a reviewed
plan for protected branches/tags and merge-request approvals.

1. Resolve the host, instance version, license tier, project visibility,
   default branch, current protections and approvals, dependency files,
   existing CODEOWNERS, and current security templates.
2. Separate local artifacts from remote settings:
   - Local: CODEOWNERS, Renovate, and selected scanning jobs.
   - Remote: protected branches/tags and approval rules are applied through
     `gitlab-ops` after review.
3. Use [codeowners-template](assets/codeowners-template) only when every
   path has a real owner. Required CODEOWNERS approval is tier-dependent.
4. Read [dependency-automation.md](dependency-automation.md) before adapting
   [renovate-template.json](assets/renovate-template.json). GitLab has no
   equivalent first-party dependency-update bot.
5. Read [approvals-protections.md](approvals-protections.md) to produce an
   exact desired-state plan and identify tier-gated enforcement.
6. Record selected guardrails and human activation steps in the harness,
   adapting [the AGENTS.md section](assets/agents-md-guardrails-section.md).

Done when: local files validate, owners and required jobs exist, the remote
plan is exact and reviewed, and version/tier limitations are explicit.
