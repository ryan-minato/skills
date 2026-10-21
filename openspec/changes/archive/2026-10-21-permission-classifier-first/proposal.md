## Why

The Claude Code settings stop for a human decision on 26 families of outward writes and allow 29 commands past the classifier. Every prompt costs the maintainer a review of an agent-written command and interrupts the run; under that load approval degrades into approving everything, so the prompts add cost without adding safety. Allow entries skip the classifier outright, which weakens safety for little saved friction, since the classifier rarely misjudges these commands. Codex has no permission configuration in this repository at all.

## What Changes

- Policy, stated once in `agent-authority.md ## Permission entries` and applied to both agents:
  - **deny** holds only commands with a clear security risk: force, mirror, all-refs, and tag pushes and remote branch deletion; merging a pull request and approving a review; any command naming `spec/approved`; `gh api` with `DELETE`, `PUT`, or `PATCH`; `gh repo delete` and `gh repo edit`; printing the `gh` token; `sudo`; broad `rm -rf`; hard resets, `git clean`, and discarding the working tree.
  - **allow** holds only the checks the workflow runs on every task: `just check`, `test`, `validate`, `lint`, `spec-validate`, `check-skill`, `spec-check`, `review-threads`, `spec-approval`, `commit-gate`. Read-only `git` and `gh` reads are left to the built-in read-only set and the classifier.
  - **ask** is empty. Everything else — pushes, pull request and issue writes, labels, workflow runs — is judged by the agent's classifier against the conversation and `AGENTS.md`, which already states that nothing reaches GitHub without the user's authorization and a publish review.
- `.claude/settings.json`: rewritten to that policy.
- `.codex/config.toml` is unchanged: Codex's approval policy, reviewer, and sandbox stay in the maintainer's global configuration.
- `.codex/rules/default.rules` (new): `forbidden` prefix rules for the deny list and `allow` rules for the same recipes; no `prompt` rule.
- `agent-authority.md ## Permission entries` and `## Threat model` restated for the new policy, including that both agents run in their classifier mode (Claude Code auto mode, Codex `auto_review`), set in the maintainer's global configuration; `harness-maintenance.md` register row and layer calibration updated; `ARCHITECTURE.md` lists `.codex/rules/`.

## Skills touched

repository change.

## Installed behavior

An agent in this repository meets a prompt only when its own classifier escalates; it is refused outright for the clearly risky commands; and the per-task checks run without review. The publish gate and the conversation's authorization stay the rule for outward writes, judged by the classifier instead of a prefix prompt.

## Impact

- Files: `.claude/settings.json`; `.codex/rules/default.rules` (new); `.agents/knowledge/agent-authority.md`, `harness-maintenance.md`; `ARCHITECTURE.md`.
- Untouched: everything under `skills/`; the workflows; the repository settings.
- The maintainer's user-level settings decide the mode: Claude Code ignores `auto` as `defaultMode` and the `autoMode` block in a project file, and Codex reads project `.codex/` files only for a trusted project.

## Non-goals

- No hook-based gate and no user-level settings committed to the repository.
- No change to the publish gate, the approval label, or who may merge.

## Deliverable and pre-authorized actions

A pull request from `chore/permission-classifier-first`. Pre-authorized by the maintainer in conversation: pushes to that branch, the draft pull request, and edits to its body.

## Tracked work

No issue.
