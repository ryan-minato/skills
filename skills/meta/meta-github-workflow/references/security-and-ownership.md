# Security, Ownership, and Dependencies

Read when the harness defines review routing, CODEOWNERS, vulnerability
reporting, scanning, secrets, deploy authority, or dependency updates.

## CODEOWNERS

One file, one chosen location — `.github/`, root, or `docs/` (first found
wins) — scoped per branch and read from the PR's **base** branch. Owners
need explicit write access; teams must be visible and have write. The last
matching pattern wins, `!` negation and character ranges are unsupported,
paths are case-sensitive, the file must stay under 3 MB, and **invalid
lines are skipped silently**. Validate exclusively through
`GET /repos/{owner}/{repo}/codeowners/errors`, never by eyeballing the
file. CODEOWNERS only *requests* review; enforcement is the ruleset's
require-code-owner-review setting — on private repositories both are
plan-gated. Derive patterns from the same real boundaries as the `area/*`
labels and the labeler map, and register the three-way edge.

Ship the errors check as a workflow: on same-repo PRs touching the file,
on push to the default branch, **and on a weekly schedule** — ownership
breaks with no commit at all (a member leaves, a team is renamed, access
is revoked), and no commit-triggered check can see that. A 404 (no file)
is a skip, not a failure; fork PRs get post-merge detection only, since
the API ref must exist in the base repository.

## Vulnerability intake

Private vulnerability reporting and repository security advisories are
**public-repository-only**. A public repository's SECURITY.md points at
PVR; a private repository's names a monitored private channel (and the
irony is worth stating: the private repo is the one that cannot receive
private reports). Never route security reports through the issue tracker.

## Scanning and dependencies

- Dependabot **alerts** are configured in settings, not in
  `dependabot.yml`; the file owns **version updates**. Generate one
  `updates:` block per ecosystem with a manifest actually present —
  including `github-actions` for the workflows this harness ships.
- Secret scanning and push protection, and code scanning/CodeQL, are free
  on public repositories; private repositories need the purchased SKUs.
  Adopt a scanner only with a triage owner, an SLA, and a false-positive
  path — an unowned alert stream is worse than none.
- Every adopted scanner or update stream is recorded in
  `platform-settings.md` with its enforcement tier and owner.

## Secrets and deploy authority

Secrets and variables are recorded by name and owner, never by value.
Prefer OIDC over long-lived cloud credentials. Environment protection
(required reviewers, wait timers) is free on public repositories and needs
a paid plan on private ones — check the probed plan rather than assuming
either way. Where it is unavailable, deploy authority is a ruleset or a
named human step, and the design says which. The design-tree decision on
which workflows may hold a secret or an OIDC identity is recorded here and
enforced by `permissions:` blocks per
[actions-and-checks.md](actions-and-checks.md).

Done when: review routing, vulnerability intake, scanning, dependency
updates, and deploy authority each name an owner and an enforcement tier,
and the CODEOWNERS file validates clean through the errors API.
