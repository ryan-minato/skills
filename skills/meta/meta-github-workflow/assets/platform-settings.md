# GitHub Platform Settings

<!-- Remote, account-side state no checkout can prove. One row per
setting; every row carries its enforcement tier and a readback. Rework
all slots; delete inapplicable rows — the three organization rows go
on a personal account, where none of those features exist. -->

Repository: `{{OWNER_REPO}}` · owner type: {{USER_OR_ORGANIZATION}} ·
visibility: {{VISIBILITY}} · plan: {{PLAN_OR_UNKNOWN}} · enforcement
tier: {{ENFORCED_ADVISORY_OR_CONVENTION}}.

| Concern | Intended state | Tier | Owner | Verify | Update trigger |
|---|---|---|---|---|---|
| Default branch & merge methods | {{VALUE}} | enforced | {{OWNER}} | `gh api repos/{{OWNER_REPO}}` | merge-policy change |
| Ruleset: {{RULESET_NAME}} | {{RULES_SUMMARY — required checks name gate jobs, approvals, linear history}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/rulesets` | job rename, policy change |
| Legacy branch protection | {{PRESENT_OR_NONE — record even when none: absence is a claim}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/branches/{{DEFAULT_BRANCH}}/protection` | ruleset edits |
| Copilot extra-approval default | {{ON_OR_OFF_AND_DECISION}} | enforced | {{OWNER}} | ruleset settings UI | agent-PR policy change |
| Actions availability & token policy | {{ENABLED_ALLOWED_ACTIONS_DEFAULT_PERMS}} | enforced | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/actions/permissions` | org policy change |
| CODEOWNERS enforcement | {{REQUIRE_CODEOWNER_REVIEW_ON_OR_OFF}} | {{TIER}} | {{OWNER}} | ruleset settings | ownership change |
| Environments & reviewers | {{ENVIRONMENTS_AND_APPROVERS}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/environments` | deploy-authority change |
| Secrets & variables (names only) | {{NAMES_AND_OWNERS}} | enforced | {{OWNER}} | `gh secret list` | rotation |
| Security features | {{DEPENDABOT_SCANNING_STATE_AND_SKUS}} | {{TIER}} | {{OWNER}} | repository security settings | plan or visibility change |
| Org issue types | {{ENABLED_TYPES_AND_WHICH_ARE_CUSTOM}} | enforced | {{OWNER}} | `gh api orgs/{{ORG}}/issue-types` | type added, renamed, or disabled |
| Org issue fields | {{FIELDS_WITH_VISIBILITY_AND_PINNED_TYPES}} | enforced | {{OWNER}} | `gh api orgs/{{ORG}}/issue-fields` | field or option change |
| Project link (only if opted in) | {{PROJECT_NUMBER_AND_URL_OR_NONE}} | {{TIER}} | {{OWNER}} | `gh project view {{PROJECT_NUMBER}} --owner {{ORG}}` | field schema or view change |

Advisory-tier upgrade trigger: {{UPGRADE_TRIGGER — e.g. "if visibility
becomes public or the plan upgrades, enable the ruleset requiring
'checks / gate' immediately"}}.

Last verified: {{TIMESTAMP_AND_EVIDENCE}}
