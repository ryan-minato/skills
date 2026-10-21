# GitHub Settings

Read this before changing or reading back any repository setting, ruleset,
security feature, Actions policy, Discussions, or label on
`ryan-minato/skills`. Every row is account-side state no checkout can
prove; this file declares the intended value of each.

Repository: `ryan-minato/skills` · personal account (no issue types, no
issue fields) · public · free plan (rulesets, scanners, and Discussions
are free on public repositories).

## Declared settings

`R` stands for `repos/ryan-minato/skills`. Each row is read with
`gh api <path> --jq '<field>'`, read-only (`\|` in a field is the table's
escape for `|`). "Current" is the readback of the date in
`## Applying and reading back`.

| Setting | Current | Intended | Path | Field |
|---|---|---|---|---|
| Default branch | `main` | `main` | `R` | `.default_branch` |
| Merge methods | rebase and squash on; merge commits off | same | `R` | `{allow_rebase_merge,allow_squash_merge,allow_merge_commit}` |
| Auto-merge | off | off | `R` | `.allow_auto_merge` |
| Delete branch on merge | on | on | `R` | `.delete_branch_on_merge` |
| Description | "Agent Skills library: install with npx skills add ryan-minato/skills" | same | `R` | `.description` |
| Discussions | on, with Q&A and Ideas (the target of `ISSUE_TEMPLATE/config.yml`) | on | `R` | `.has_discussions` |
| Projects, Wiki | off | off | `R` | `{has_projects,has_wiki}` |
| Ruleset `Default` | id 19602018, active, targets the default branch | same | `R/rulesets/19602018` | `{enforcement,conditions}` |
| Ruleset rules | deletion and non-fast-forward blocked; pull request required | same | `R/rulesets/19602018` | `[.rules[].type]` |
| Required checks | `checks / gate`, `pr / policy`, `scan-secrets`; strict | the same and `checks / archived` | `R/rulesets/19602018` | `.rules[]\|select(.type=="required_status_checks").parameters` |
| Pull request rule | 0 approvals; threads resolved; merge methods squash, rebase | same | `R/rulesets/19602018` | `.rules[]\|select(.type=="pull_request").parameters` |
| Ruleset bypass | none | none | `R/rulesets/19602018` | `.bypass_actors` |
| Legacy branch protection | none (HTTP 404) | none | `R/branches/main/protection` | HTTP 404 means none |
| Actions allowed | all | GitHub-owned plus the allowlist below | `R/actions/permissions` | `.allowed_actions` |
| Actions allowlist | none (all allowed) | `denoland/setup-deno@*`, `trufflesecurity/trufflehog@*`, `devcontainers/ci@*` | `R/actions/permissions/selected-actions` | `.patterns_allowed` |
| Full-length SHA pinning | not required | required | `R/actions/permissions` | `.sha_pinning_required` |
| Default workflow permissions | read; Actions cannot approve reviews | same | `R/actions/permissions/workflow` | `.` |
| Secret scanning | on | on | `R` | `.security_and_analysis.secret_scanning.status` |
| Push protection | on | on | `R` | `.security_and_analysis.secret_scanning_push_protection.status` |
| Dependabot alerts | on | on | `R/vulnerability-alerts` | HTTP 204 means on, 404 off |
| Dependabot security updates | on | on | `R` | `.security_and_analysis.dependabot_security_updates.status` |
| Private vulnerability reporting | off | on (`.github/SECURITY.md` points at it) | `R/private-vulnerability-reporting` | `.enabled` |
| Code scanning | CodeQL default setup, `actions` and `python`; advisory | same | `R/code-scanning/default-setup` | `{state,languages}` |
| Labels | exactly `.github/labels.json` | exactly `.github/labels.json`, applied by `labels / sync` | `R/labels?per_page=100` | `[.[].name]` |
| Actions secrets | none; workflows use `GITHUB_TOKEN` | none | `R/actions/secrets` | `.total_count` |

- The allowlist is every third-party action a workflow or
  `.github/actions/*/action.yml` uses; `actions/*` and `github/*` are
  GitHub-owned. A new third-party action adds its row value here in the
  change that adds the `uses:` line.
- 0 required approvals: a solo maintainer cannot approve their own pull
  request; the integration gate in `.agents/knowledge/agent-authority.md`
  is the human decision.
- No bypass actor: no workflow pushes anywhere, so no identity needs one.
- Scanner alerts (secret scanning, CodeQL, Dependabot) and private
  vulnerability reports are triaged by the maintainer; a false positive is
  dismissed with a reason on the alert.

## Applying and reading back

- An agent never writes a setting (`.agents/knowledge/agent-authority.md`).
  The maintainer applies every row whose Current differs from Intended,
  after the change that declared it is archived.
- The agent then reads back every row with its command and records one
  dated result — which rows match, which differ — in that change's pull
  request description, since a commit after the archive would spend the
  approval. The next change copies the result into the Current column
  and the line below.
- The periodic review reads back every row too
  (`.agents/knowledge/harness-maintenance.md` `## Periodic entropy review`).

Last readback: 2026-10-20, before ready of `repo-harness-alignment`.
Differ: Actions allowed, Actions allowlist, full-length SHA pinning,
private vulnerability reporting. Every other row matches.

## Update this file when

- A setting, the ruleset, or a security feature is intended to change: the
  row changes in the same pull request as the reason for it.
- A required check is renamed (`.agents/knowledge/github-checks.md`).
- A workflow adds or drops a third-party action.
- A readback differs from the declaration.
