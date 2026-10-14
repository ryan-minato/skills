# GitHub Platform Settings

<!-- Remote, account-side state no checkout can prove. One row per
setting; every row carries its enforcement tier and a readback. Rework
all slots; delete inapplicable rows — the three organization rows go
on a personal account, where none of those features exist. -->

Repository: `{{OWNER_REPO}}` · owner type: {{USER_OR_ORGANIZATION}} ·
visibility: {{VISIBILITY}} · plan: {{PLAN_OR_UNKNOWN}} · enforcement
tier: {{ENFORCED_ADVISORY_OR_CONVENTION}}.

| Concern | Intended state | Tier | Owner | Verify | Update trigger | Described by |
|---|---|---|---|---|---|---|
| Default branch & merge methods | {{VALUE}} | enforced | {{OWNER}} | `gh api repos/{{OWNER_REPO}}` | merge-policy change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Ruleset: {{RULESET_NAME}} | {{RULES_SUMMARY — PR required, force pushes blocked, deletions restricted, required checks name gate jobs}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/rulesets` | job rename, policy change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Ruleset bypass actors | {{WHO_MAY_BYPASS_AND_WHY — empty means admins cannot push directly either}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/rulesets` | membership or policy change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Legacy branch protection | {{PRESENT_OR_NONE — record even when none: absence is a claim}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/branches/{{DEFAULT_BRANCH}}/protection` | ruleset edits | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Copilot extra-approval default | {{ON_OR_OFF_AND_DECISION}} | enforced | {{OWNER}} | ruleset settings UI | agent-PR policy change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Actions availability & token policy | {{ENABLED_ALLOWED_ACTIONS_DEFAULT_PERMS}} | enforced | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/actions/permissions` | org policy change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| CODEOWNERS enforcement | {{REQUIRE_CODEOWNER_REVIEW_ON_OR_OFF}} | {{TIER}} | {{OWNER}} | ruleset settings | ownership change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Environments & reviewers | {{ENVIRONMENTS_AND_APPROVERS}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/environments` | deploy-authority change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Configuration repository collaborators (configuration repository only) | organization owners only — a pull request runs its own code under the read-only plan token | enforced | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/collaborators` | membership change | none |
| Secrets & variables (names only) | {{NAMES_AND_OWNERS}} | enforced | {{OWNER}} | `gh secret list` | rotation | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Secret scanning | {{ENABLED_OR_DOWNGRADED_WITH_SKU}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}} --jq .security_and_analysis` | plan or visibility change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Push protection | {{ENABLED_OR_DOWNGRADED_WITH_SKU}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}} --jq .security_and_analysis` | plan or visibility change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Code scanning | {{DEFAULT_SETUP_STATE_AND_LANGUAGES_OR_ADVANCED}} | {{TIER}} | {{OWNER}} | `gh api repos/{{OWNER_REPO}}/code-scanning/default-setup` | language added, plan change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Dependabot | {{ALERTS_AND_VERSION_UPDATE_STATE}} | {{TIER}} | {{OWNER}} | repository security settings | ecosystem added | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Org issue types | {{ENABLED_TYPES_AND_WHICH_ARE_CUSTOM}} | enforced | {{OWNER}} | `gh api orgs/{{ORG}}/issue-types` | type added, renamed, or disabled | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Org issue fields | {{FIELDS_WITH_VISIBILITY_AND_PINNED_TYPES}} | enforced | {{OWNER}} | `gh api orgs/{{ORG}}/issue-fields` | field or option change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |
| Project link (only if opted in) | {{PROJECT_NUMBER_AND_URL_OR_NONE}} | {{TIER}} | {{OWNER}} | `gh project view {{PROJECT_NUMBER}} --owner {{ORG}}` | field schema or view change | {{POLICY_TF_CONFIG_REPO_OR_NONE}} |

"Described by" names the file that declares the row: `.github/policy.tf`,
the organization's configuration repository, or `none` when no provider
covers it (the readback is then the only check). A row a tf file
describes is verified by a plan, not only by its readback.

## Harness options

| Option | Choice | Default | Recorded on |
|---|---|---|---|
| Platform settings as tf | {{YES_IF_TOFU_OR_TERRAFORM_INSTALLED_OR_NO}} | yes when the tool is installed | {{DATE}} |
| Label sync | {{CI_OR_MANUAL}} | CI (`labels / sync`, `issues: write`) | {{DATE}} |
| Label pruning | {{OFF_OR_ON}} | off — deleting a label strips it from every issue and pull request | {{DATE}} |
| Configuration repository sync (configuration repository only) | {{CI_PLAN_AND_APPLY_OR_CI_PLAN_LOCAL_APPLY_OR_LOCAL}} | CI plans and applies | {{DATE}} |

Advisory-tier upgrade trigger: {{UPGRADE_TRIGGER — e.g. "if visibility
becomes public or the plan upgrades, enable the ruleset requiring
'checks / gate' immediately"}}.

Last verified: {{TIMESTAMP_AND_EVIDENCE}}
