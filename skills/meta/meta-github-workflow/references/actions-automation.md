# Community and Contract Automation

Read when the repository takes issues or pull requests from anyone other
than the sole author, or when the harness ships a taxonomy, template,
commit, or tag contract that only a workflow can enforce.

## What ships versus what stays a recipe

Ship a workflow file when its mechanics are fragile enough that
regenerating it from memory would produce a subtly broken version;
otherwise leave a recipe here. Shipped by default when this branch is
selected: the PR checklist check, the commit check, the tag check, the PR
labeler, the triage lifecycle, and the taxonomy check (all under
the `assets/` directory). Recipes only, adopt on explicit request:

- **Stale automation** — pure policy that enforces no harness contract,
  and closing other people's issues on a timer is the most resented
  automation in open source. If adopted: generous windows, exempt labels,
  and a named owner.
- **Issue regex labeler** — heuristic guessing on contributor-facing
  objects; the forms' `labels:` key already labels deterministically. Only
  for high-volume repositories that must keep blank issues enabled.
- **CODEOWNERS errors check** — specified in
  [security-and-ownership.md](security-and-ownership.md); wire it here.
- **Auto-assign and review rotation** — org-level team settings do this
  natively; prefer the platform setting.

## Fork-safe design rules

- PR-side policy is expressed as **failing checks** that read the event
  payload and `core.setFailed` — fork PRs carry read-only tokens, so a
  mutating workflow simply fails there. Issue-side automation is
  unaffected: issues have no fork concept and always run with a writable
  token.
- Label mutation on fork PRs requires `pull_request_target`; treat it as a
  reviewed exception under the rules in
  [actions-and-checks.md](actions-and-checks.md) — the shipped labeler
  defaults to plain `pull_request` and is escalated only when fork PRs are
  evidenced, and only because the pinned labeler reads its config via the
  API without a checkout.
- Detection, not prevention: Actions runs after the state exists. The
  triage workflow **comments** on a second `priority/*` rather than
  silently removing it, and uses a per-issue concurrency group with
  `cancel-in-progress: false`.
- Automation uses `GITHUB_TOKEN` only — its events do not retrigger
  workflows, so recursion never starts. A PAT or App token reopens that
  door; do not introduce one for labeling.

## Contract enforcement mapping

| Harness contract | Enforcer |
|---|---|
| PR template checklist completed | checklist workflow — parses the template's own `## ` headings, fail-closed, `types: [opened, edited, synchronize]` |
| Commit or PR-title convention | commit-check workflow running the committed validator (title mode under squash; range mode with `fetch-depth: 0` otherwise) |
| Tag format on release tags | tag-check workflow reading the committed versioning config |
| Labels agree across `labels.json`, `release.yml`, forms, labeler | taxonomy check — static parse, fork-safe, required-check eligible |
| `area/*` labels track changed paths | PR labeler + `labeler.yml`, sharing its path map with CODEOWNERS |
| Triage lifecycle (`status/needs-triage` on/off, one priority) | triage workflow — issue events only, comment-not-remove |

Every automated comment or failure message names the file to edit. Scale
by the proportionality rule in [durable-harness.md](durable-harness.md):
none of this ships to a solo repository by default.

Done when: each shipped workflow maps to a contract the harness actually
created, carries explicit `permissions:`, appears in the job-name registry,
and has its healthy-run shape recorded in `checks.md`.
