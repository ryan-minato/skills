---
name: change-workflow
description: Repository change workflow — keeps tracked work on a dedicated branch, uses atomic commits and validation gates, and opens a draft PR before review. Use when tracked files are about to change; when creating or resuming a work branch; when preparing commits, pushing, opening a PR, or handing unfinished work to another agent. No external issue tracker is required.
metadata:
  internal: true
---

# Change Workflow

Follow this for every tracked-file change. Git branches, commits, and the pull
request are the durable work record; this repository does not require an
external project-management issue.

## 1. Establish the branch

Read-only analysis needs no branch. Before the first tracked edit:

1. Confirm the worktree is clean enough to isolate the requested change.
2. Fetch `origin` when network access is available.
3. Create or resume one descriptive branch from current `origin/main`.
4. Never work directly on `main`.

Preserve unrelated user changes and never reset them away.

## 2. Work in coherent milestones

Group work into independently reviewable milestones. After each milestone,
record what is done, what remains, deviations, and validation in the current
handoff or PR description. Do not create an external issue merely to satisfy
process.

## 3. Commit atomically

Every commit must be independently valid:

1. Stage one logical change.
2. Run `just commit-gate`.
3. Apply the `sensitivity-check` skill to staged content, checking secrets and
   PII; a confirmed finding blocks the commit.
4. Complete the `git-commit` skill gates and use the repository's
   Conventional Commit convention.
5. Never bypass hooks and never add tool-attribution trailers.

Use a GitHub or GitLab anonymous author email unless the user explicitly
approves a private address.

## 4. Push and open a draft PR

After the first coherent, validated implementation:

1. Run `just check` and fix every failure.
2. Push the branch without force.
3. Open a draft PR with a Conventional Commit title and the repository PR
   template.
4. Keep work and additional commits on the same branch.

## 5. Ready for review

When all requested scope is implemented:

1. Run `just check` again.
2. Push the final branch state.
3. Update the PR body with the final change summary, validation, deviations,
   and follow-ups.
4. Mark the draft ready for human review.

## Gotchas

- A branch or PR does not authorize unrelated cleanup.
- Do not invent a ticket, milestone, or external tracker entry.
- If unrelated work is discovered, report it separately instead of expanding
  the current branch.
- When ending with work unfinished, leave a self-contained handoff in the PR
  or conversation: branch, completed work, remaining work, and current checks.
