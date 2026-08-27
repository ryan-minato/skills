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

## Scanning: default on, plan-gated

Three scanners are part of the default baseline, proposed on every build
rather than offered as options: **secret scanning**, **push protection**
(which rejects a push whose commits carry a recognized secret), and
**code scanning** through CodeQL default setup.

They are free on **public** repositories. On private or internal ones they
need separately purchased SKUs — GitHub Secret Protection for the first
two, GitHub Code Security for the third. **A paid plan does not include
them**: a private repository on Team has no scanning until the SKU is
bought, which is the assumption that most often turns this baseline into a
promise the repository cannot keep. Downgrade it in writing with its
upgrade trigger, exactly as the ruleset is in
[rules-and-protection.md](rules-and-protection.md).

Enable and read back through the repository object: secret scanning and
push protection are the `security_and_analysis` keys `secret_scanning` and
`secret_scanning_push_protection` (each `enabled` or `disabled`) on
`PATCH /repos/{owner}/{repo}`; code scanning has its own endpoint,
`PATCH /repos/{owner}/{repo}/code-scanning/default-setup`. Default setup
needs Actions enabled and a CodeQL-supported language — where the language
is unsupported, say so and propose advanced setup rather than reporting
coverage the repository does not have.

The owner requirement stands and is part of the baseline, not an
alternative to it: every enabled scanner names a triage owner, an SLA, and
a false-positive path. An unowned alert stream is worse than none, so a
scanner whose owner is still undecided appears in the plan with that gap
named — never silently dropped.

- Dependabot **alerts** are configured in settings, not in
  `dependabot.yml`; the file owns **version updates**. Generate one
  `updates:` block per ecosystem with a manifest actually present —
  including `github-actions` for the workflows this harness ships.
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

Done when: review routing, vulnerability intake, dependency updates, and
deploy authority each name an owner and an enforcement tier; each baseline
scanner is either enabled and read back or downgraded in writing with its
SKU and upgrade trigger; and the CODEOWNERS file validates clean through
the errors API.
