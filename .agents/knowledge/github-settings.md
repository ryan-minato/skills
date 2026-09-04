# GitHub Settings Register

Read this before changing any repository setting, ruleset, security
feature, Actions policy, Discussions, or label on `ryan-minato/skills`.
Every row is account-side state no checkout can prove; changing one is a
human-approved remote write, executed then read back and recorded here.

Repository: `ryan-minato/skills` · owner type: personal account (no issue
types, no issue fields) · visibility: public · plan: free features suffice
(rulesets, scanners, Discussions are free on public repositories).

| Concern | Intended state | Tier | Verify | Update trigger |
|---|---|---|---|---|
| Default branch | `main` | enforced | `gh api repos/ryan-minato/skills --jq .default_branch` | never expected |
| Merge methods | rebase and squash allowed; merge commits disabled | enforced | `gh api repos/ryan-minato/skills --jq '{allow_rebase_merge,allow_squash_merge,allow_merge_commit}'` | merge policy in `github-workflow.md` changes |
| Delete branch on merge | enabled | enforced | `gh api repos/ryan-minato/skills --jq .delete_branch_on_merge` | branch policy changes |
| Description | "Agent Skills library: install with npx skills add ryan-minato/skills" | convention | `gh repo view --json description` | purpose changes |
| Ruleset `Default` (id 19602018) | targets `main`; pull request required; deletion and force push blocked; required checks `checks / gate`, `pr / policy`, `scan-secrets` (strict); review threads must be resolved; 0 required approvals; extra approval for unattributed changes off; allowed merge methods squash and rebase | enforced | `gh api repos/ryan-minato/skills/rulesets/19602018` | a check is renamed; approval policy changes |
| Ruleset bypass actors | **pending maintainer action**: the GitHub Actions app, so `spec-archive.yml` can push its archive commit to `main`; nobody else (the maintainer also goes through pull requests). Until granted, changes are archived inside the pull request (`spec-workflow.md`, Archive mode) | enforced | same readback (`bypass_actors`) | the archive workflow is added, renamed, or removed; a second maintainer joins |
| Required approvals | 0 — a solo maintainer cannot approve their own pull request; the integration gate in `agent-authority.md` is the human decision | convention | same readback | a second maintainer joins |
| Legacy branch protection | none | enforced | `gh api repos/ryan-minato/skills/branches/main/protection` (404 = none) | ruleset edits |
| Actions | enabled; all actions allowed by policy, but every workflow pins by commit SHA and the only third-party action is TruffleHog | enforced where GitHub can | `gh api repos/ryan-minato/skills/actions/permissions` | a new action is added |
| Secret scanning | enabled | enforced | `gh api repos/ryan-minato/skills --jq .security_and_analysis.secret_scanning` | visibility change |
| Push protection | enabled | enforced | `gh api repos/ryan-minato/skills --jq .security_and_analysis.secret_scanning_push_protection` | visibility change |
| Code scanning | CodeQL default setup, Python and Actions; advisory (not a required check); alerts triaged by the maintainer | advisory | `gh api repos/ryan-minato/skills/code-scanning/default-setup` | a language is added |
| Dependabot | version updates for `github-actions` and `devcontainers`, weekly (`.github/dependabot.yml`); alerts enabled | enforced | repository security settings | a manifest is added |
| Discussions | enabled, with Q&A and Ideas categories; the target of `ISSUE_TEMPLATE/config.yml` | enforced | `gh repo view --json hasDiscussionsEnabled` | intake routing changes |
| Projects, Wiki | disabled (unused) | convention | `gh repo view --json hasProjectsEnabled,hasWikiEnabled` | a "not used" trigger in `github-workflow.md` fires |
| Labels | exactly `.github/labels.json`; applied with `python3 scripts/sync_labels.py --file .github/labels.json --repo ryan-minato/skills` (dry run, then `--apply`); `--prune` only after listing the issues that carry the label and with explicit authorization | enforced | `gh label list` | `labels.json` changes |
| Milestones | created or confirmed by the maintainer only; closed, never deleted | convention | `gh api repos/ryan-minato/skills/milestones` | — |
| Private vulnerability reporting | enabled; `SECURITY.md` points at it | enforced | repository security settings | — |
| Secrets and variables | none; every workflow uses `GITHUB_TOKEN` with job-level permissions | enforced | `gh secret list` | a workflow needs a credential (a maintainer decision) |

## Ownership and security

- Vulnerability intake: private vulnerability reporting; triage owner the
  maintainer; acknowledgement within seven days (`.github/SECURITY.md`).
- Scanner alerts (secret scanning, CodeQL, Dependabot): the maintainer
  reviews them when they arrive; a false positive is dismissed with a
  reason on the alert.
- No CODEOWNERS and no required reviewer: solo maintainer. Trigger: a
  second maintainer joins.
- No releases: consumers install from `main`. Trigger: consumers depend on
  named versions.

## Last verification

2026-09-03, after the authorized writes of the harness rebuild, read back
with the commands in the table: description set; Discussions on (default
categories including Q&A and Ideas); Projects and Wiki off; merge commits
off, rebase and squash on; delete branch on merge on; labels exactly
`.github/labels.json` (the three dead labels pruned after confirming no
issue or pull request carried them); secret scanning, push protection,
and CodeQL default setup on; no legacy branch protection. Still differing
from the intended state until `checks / gate` is green on `main` and the
write is authorized: ruleset `Default` (required checks still only
`scan-secrets`, thread resolution off, extra approval for unattributed
changes on, all merge methods listed).
