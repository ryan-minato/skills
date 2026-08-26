# Rulesets and Merge Protection

Read when the harness enforces anything at merge, push, or tag time.

## The enforcement tier comes first

What this file can deliver is set by the capability quadrant: on a private
Free-plan repository there are no rulesets and no branch protection —
**"required check" does not exist there**. In that tier this branch
produces a written convention plus a named human, labeled "convention" in
those words, and a pre-computed upgrade trigger ("if this becomes public or
the plan upgrades, enable the ruleset with these exact job names"). Never
ship a plan that silently no-ops.

## Two layers, both live

Rulesets are current: up to 75 per repository, multiple rulesets aggregate
with most-restrictive-wins, anyone with read access can view active ones
(agents can discover their own constraints without admin), and an
Active/Disabled toggle allows safe rollout. Legacy branch protection still
applies underneath and layers with them — a stale classic rule silently
tightens a ruleset. Inventory **both** layers before concluding a rule is
missing or excessive, and record both in `platform-settings.md`; any
"why is this blocked?" procedure checks both.

## Designing the ruleset

- Require a pull request before merging, with the approval count the team
  agreed; note the **unattributed-Copilot extra approval is enabled by
  default** and turns one required approval into two for agent-authored
  PRs — decide it explicitly.
- Required status checks name **jobs from the registry** — the aggregator
  gate where path filters exist, never a filtered job (skipped = Success).
  A required check whose producing job was renamed blocks every PR
  forever; that synchronization edge is registered in the deposit.
- Sub-settings to decide deliberately: dismiss stale approvals, require
  code-owner review (this is what makes CODEOWNERS enforcement real),
  require conversation resolution, require linear history, block force
  pushes, restrict deletions, required merge method.
- Tag rulesets protect release tags; pair them with the tag-check workflow
  so format and protection agree.
- Bypass actors are part of the design, not an afterthought: name who may
  bypass and why, and record it.
- Merge queue's ruleset support is documented ambiguously — verify on the
  live repository before designing on it; its workflow-side demands live
  in [actions-and-checks.md](actions-and-checks.md).

## Apply and audit

Rule changes are remote writes under the SKILL.md sequence: read current
state, draft the exact delta, confirm plan and permission preconditions,
state the rollback, apply with approval, read back. Report tier-gated or
permission-blocked fields plainly instead of retrying broader scopes.

Done when: every rule the design promises is either platform-enforced and
read back, or recorded as advisory/convention with its owner and upgrade
trigger — and no required check names a job that does not exist.
