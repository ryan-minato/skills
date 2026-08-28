# Rulesets and Merge Protection

Read when the harness enforces anything at merge, push, or tag time.

## The baseline, then the tier

The default posture is a **protected default branch**: merging goes
through a pull request, and nothing reaches the branch by direct push.
Propose it on every build. It is not a preference to elicit from a blank
slate — the user subtracts from it and says why, rather than assembling
protection from nothing.

What the platform will actually accept is set by the capability quadrant:
on a private Free-plan repository there are no rulesets and no branch
protection — **"required check" does not exist there**. In that tier the
baseline is not dropped, it is *downgraded in writing*: a written
convention plus a named human, labeled "convention" in those words, and a
pre-computed upgrade trigger ("if this becomes public or the plan
upgrades, enable the ruleset with these exact job names"). Never ship a
plan that silently no-ops, and never let an unavailable rule disappear
from the plan instead of appearing as a downgrade.

## The default baseline ruleset

Target the default branch, and read the current state of **both**
protection layers before proposing a delta:

- Require a pull request before merging. **There is no "block direct
  push" switch** — this rule is what produces that effect, and only when
  bypass actors do not hand it back. A repository admin bypasses by
  default unless the bypass list says otherwise, so name the bypass actors
  explicitly or accept that "must go through a PR" is untrue for admins.
- Block force pushes and restrict deletions.
- Require status checks, naming the aggregator gate job.
- Require conversation resolution.

Everything past this line — approval counts, code-owner review, linear
history, dismissing stale approvals, required merge method — is a design
decision from the frontier, not part of the baseline.

## Two layers, both live

Rulesets are current: up to 75 per repository, multiple rulesets aggregate
with most-restrictive-wins, anyone with read access can view active ones
(agents can discover their own constraints without admin), and an
Active/Disabled toggle allows safe rollout. Legacy branch protection still
applies underneath and layers with them — a stale classic rule silently
tightens a ruleset. Inventory **both** layers before concluding a rule is
missing or excessive, and record both in `platform-settings.md`; any
"why is this blocked?" procedure checks both.

## Designing past the baseline

- The approval count is the team's decision; note the **unattributed-Copilot
  extra approval is enabled by default** and turns one required approval
  into two for agent-authored PRs — decide it explicitly.
- The baseline's required check names a **job from the registry** — the
  aggregator gate where path filters exist, never a filtered job
  (skipped = Success). A required check whose producing job was renamed
  blocks every PR forever; that synchronization edge is registered in the
  deposit.
- Sub-settings to decide deliberately: dismiss stale approvals, require
  code-owner review (this is what makes CODEOWNERS enforcement real),
  require linear history, required merge method.
- Tag rulesets protect release tags; pair them with the tag-check workflow
  so format and protection agree.
- Merge queue's ruleset support is documented ambiguously — verify on the
  live repository before designing on it; its workflow-side demands live
  in [actions-and-checks.md](actions-and-checks.md).

## Apply and audit

Rule changes are remote writes under the SKILL.md sequence: read current
state, draft the exact delta, confirm plan and permission preconditions,
state the rollback, apply with approval, read back. Report tier-gated or
permission-blocked fields plainly instead of retrying broader scopes.

Done when: every baseline rule is either platform-enforced and read back,
or recorded as advisory/convention with its owner and upgrade trigger; no
required check names a job that does not exist; and the bypass actors are
named, so "must go through a pull request" is either true or qualified.
