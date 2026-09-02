# GitHub Platform Settings

Read this before changing GitHub repository settings, merge policy, security
features, Actions permissions, or required checks.

Repository: `ryan-minato/skills`; owner type: personal account; visibility:
public; plan: unknown; maintainer: `ryan-minato`.

| Concern | Intended state | Tier | Verify | Update trigger |
|---|---|---|---|---|
| Default branch | `main` | Enforced | `gh api repos/ryan-minato/skills --jq .default_branch` | Default-branch change |
| Merge methods | Rebase for same-repository PRs; squash for fork PRs; merge commits disabled | Pending remote sync | `gh api repos/ryan-minato/skills --jq '{allow_rebase_merge,allow_squash_merge,allow_merge_commit}'` | Merge-policy change |
| Branch cleanup | Delete same-repository branches after merge; never delete existing branches as part of enabling it | Pending remote sync | `gh api repos/ryan-minato/skills --jq .delete_branch_on_merge` | Branch-lifecycle change |
| Ruleset `Default` | Pull request required; deletion and force push blocked; strict required checks are `scan-secrets`, `checks / quality`, and `pr / policy`; conversations resolved | Pending remote sync | `gh api repos/ryan-minato/skills/rulesets/19602018` | Check rename or acceptance-policy change | <!-- pragma: allowlist secret -->
| Ruleset bypass | No bypass actors | Enforced | Same ruleset readback | Maintainer or policy change |
| Required approvals | Zero; the H1 policy keeps integration human-owned without deadlocking maintainer-authored PRs | Convention | Same ruleset readback | Contributor or identity model changes |
| Legacy protection | None expected; current access cannot distinguish absent from unreadable | Pending verification | `gh api repos/ryan-minato/skills/branches/main/protection` or repository settings | Ruleset change |
| Actions policy | Enabled; explicit least-privilege workflow permissions; official actions pinned to commits except the recorded secret-workflow exception | Enforced where visible | `gh api repos/ryan-minato/skills/actions/permissions` or repository settings | Actions policy change |
| Secret scanning | Enabled | Advisory security service | `gh api repos/ryan-minato/skills --jq .security_and_analysis.secret_scanning` or repository settings | Visibility or security-policy change |
| Push protection | Enabled | Preventive security control | `gh api repos/ryan-minato/skills --jq .security_and_analysis.secret_scanning_push_protection` or repository settings | Visibility or security-policy change |
| Code scanning | Default setup for Python and Actions; not a required check | Advisory | `gh api repos/ryan-minato/skills/code-scanning/default-setup` | Language or security-policy change |
| Dependabot | No version-update automation | Convention | Repository security settings | Automated dependency PRs are approved |
| Auto-merge and merge queue | Disabled | Enforced | Repository settings and ruleset | Integration policy changes |

Conditional merge methods cannot be represented by a GitHub ruleset. The
repository therefore enables both rebase and squash, disables merge commits,
and relies on the maintainer and `change-workflow` to choose the correct
method.

Last read-only verification: 2026-09-02. The API confirmed a public personal
repository, `main`, one active ruleset with no bypass actors, strict
`scan-secrets`, CodeQL runs, and rebase/squash/merge currently enabled. Merge
commit disablement, branch cleanup, the two new required checks, and
conversation resolution still await authorized remote synchronization. Admin
readback is also required for Actions policy, legacy protection, secret
scanning, push protection, and CodeQL configuration.
