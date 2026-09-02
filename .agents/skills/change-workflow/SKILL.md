---
name: change-workflow
description: GitHub-native repository change workflow — verifies gh and explicit remote authorization, uses optional Issues, dedicated branches, atomic commits, and a draft-to-ready PR lifecycle. Use when tracked files are about to change; when starting or resuming repository work; when preparing commits, pushing, opening or readying a PR; or when work needs a durable handoff. Read-only analysis does not trigger it.
metadata:
  internal: true
---

# Change Workflow

Follow this for every tracked-file change. GitHub Issues are optional; the work
branch and pull request are the durable record for every change.

## 1. Resolve the governing contracts

Read `.agents/knowledge/project-workflow.md` before starting the change and
`.agents/knowledge/agent-authority.md` before any review-admission or
integration action. H1 applies only after the goal, scope, non-goals,
constraints, and acceptance criteria are complete. If they are incomplete,
resolve them with the user before treating the work as autonomously authored.

The authority contract is policy even when GitHub or an available token would
permit more. Never edit either contract, a required check, or repository
protection to unblock your own change.

## 2. Gate GitHub tooling and authorization

Run these checks before any other workflow step:

```bash
command -v gh
gh --version
gh auth status
```

If `gh` is missing, stop and direct the user to
<https://cli.github.com/>. Offer `brew install gh` on macOS,
`winget install --id GitHub.cli` on Windows, and the official Linux package
instructions from that page. If authentication fails, stop and ask the user to
run `gh auth login`; never start the interactive login for them.

Then identify the remote writes the task may require: labels or Issues, pushing
a branch, creating or editing a PR, and marking it ready. Continue only when the
user explicitly authorized those operations in the current conversation. A
clear authorization for the whole workflow covers its listed operations; do not
re-ask at each step. If authorization is absent or ambiguous, make no workflow
changes and wait for clarification. An explicit local-only instruction permits
local branch and commit work but no remote writes.

Done when: `gh auth status` succeeds and the allowed remote operations are
recorded, or the workflow has stopped with exact setup or authorization needs.

## 3. Resolve optional Issue context

Derive `OWNER/REPO` from `git remote get-url origin`. If the user supplied an
Issue, verify it with `gh issue view` and use it. Otherwise search open Issues
and PRs for overlapping work before creating anything. Create an Issue only
when the user explicitly requested or authorized Issue creation; the absence of
an Issue never blocks an otherwise authorized change.

Apply only labels that exist in `.github/labels.json`. Use one `priority/*`
label and every applicable `catalog/*` label. For work outside a public catalog,
use `catalog/repository`.

Record either the verified Issue number or a short reason why the change has no
Issue. Never invent an identifier.

## 4. Establish the branch

Before the first tracked edit:

1. Confirm the worktree is clean enough to isolate the requested change.
2. Fetch `origin` when remote reads are allowed.
3. Create or resume one branch from current `origin/main`.
4. With an Issue, name it `<type>/<issue-number>-<slug>`; without one, use
   `<type>/<slug>`.
5. Never work directly on `main`.

Use the dominant Conventional Commit type for `<type>`. Preserve unrelated user
changes and never reset them away.

## 5. Work in atomic milestones

Each commit is the smallest independently valid logical change:

1. Group files that must land together for checks or quality requirements to
   pass; splitting them would create an invalid intermediate state.
2. Split changes that can independently meet the repository quality bar.
3. Stage only one resulting group and run its applicable tests.
4. Run `just commit-gate`, then apply the `sensitivity-check` skill to the
   staged content. A confirmed secret or PII finding blocks the commit.
5. Complete every `git-commit` skill gate and use the repository Conventional
   Commit convention.

Never bypass hooks, force a failing commit, or add tool-attribution trailers.
Use a GitHub or GitLab anonymous author email unless the user explicitly
approves a private address.

After each milestone, record completed work, remaining work, deviations, and
validation in the draft PR or current handoff.

## 6. Publish the draft PR

Open the draft after a coherent baseline implementation works and passes its
applicable checks, before optional refinement and the final test pass:

1. Run `just check` and fix every failure.
2. Push the branch without force.
3. Open a draft PR with a Conventional Commit title and
   `.github/PULL_REQUEST_TEMPLATE.md`.
4. Put `Closes #N` in `Related issue` when an Issue exists. Otherwise put
   `N/A — <reason>`.

The draft is the public ownership signal that prevents duplicate effort; it is
not a claim that review can begin. Keep improvements, tests, and additional
atomic commits on the same branch and update the PR body as evidence changes.

## 7. Ready for review

Mark the PR ready only when all requested scope and refinement are complete:

1. Run `just check` again.
2. Complete every task-specific test, including behavioral tests required for
   changed Skills.
3. Push the final branch state and wait for GitHub checks to finish.
4. Update the PR body with the final summary, validation, deviations, and
   follow-ups; tick every required checklist item.
5. Mark the draft ready for human review. Never merge it automatically.

## 8. Hand off integration

The human maintainer owns the integration decision and performs the merge.
Use rebase merge for a branch in `ryan-minato/skills`; use squash merge for a
fork pull request. GitHub cannot enforce this distinction by pull-request
origin, so report the required method explicitly in the handoff. Merge commits
are not allowed.

Every readiness or escalation handoff must make the authority boundary
decision-ready. State which authorized H1 author actions remain available
(push, update the draft, mark ready, and request review), that the agent will
not approve or merge, and the required integration method. Also state that the
agent will not edit authority policy, required checks, or protection rules to
unblock the change. If a required check is unavailable or unreliable, identify
it as an escalation condition rather than presenting the change as ready.
Include the goal, tests and CI state, actual scope and deviations, risks and
limitations, and the human maintainer's available decisions.

## Gotchas

- GitHub authorization is task-scoped; a successful `gh auth status` is not
  permission to publish.
- A complete specification permits H1 review admission; it never permits an
  agent to approve or merge.
- Issue closing keywords take effect on merge, not when the PR becomes ready.
- A branch, Issue, or PR does not authorize unrelated cleanup.
- If unrelated work is discovered, report it separately instead of expanding
  the current branch.
- When ending with work unfinished, leave a self-contained handoff in the PR
  or conversation: branch, completed work, remaining work, and current checks.
