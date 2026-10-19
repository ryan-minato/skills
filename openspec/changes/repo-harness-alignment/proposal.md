## Why

This repository's harness was read against the same harness specification and the same two deposited harnesses that drive the `harness-methodology-alignment` and `spec-contract-alignment` changes. Most of it holds: the entrypoint is two-part and routes by situation, every pointer and command resolves and `scripts/validate_harness.py` keeps it so, secret scanning runs on every commit, the specification check blocks the merge, and the authority policy governs the management actions. What it lacks is the execution-level side of the same layers: the framework permission file allows and never denies, so a mistaken force push or a `gh pr merge` is stopped by nothing but prose; no rule tells an agent blocked mid-task to continue and queue the decision, or where a found harness mismatch must land; no checkpoint or squash rule exists beside the atomic-commit rule; the dev container leaves pagers on for `gh` and `git`; the workflow security rules (explicit permissions, SHA-pinned actions, bounded `pull_request_target`) are enforced by review only; and the six-layer calibration this harness implies is written nowhere, so a future recalibration starts from scratch.

## What Changes

- `.claude/settings.json`: a `deny` list for known-dangerous commands (force push, hard reset, branch deletion of `main`, `gh pr merge`, `gh api` deletes, recursive removal outside the scratch directory) with a comment naming the sandbox as the security boundary and the list as fool-proofing; the `allow` list extended with the read-only and recipe commands the change workflow runs on every task (`just spec-check`, `just spec-changes`, `openspec new change`, `gh api` reads).
- `AGENTS.md ## Always`: two rules — blocked work continues on everything that does not depend on the decision and queues the item in the handoff's pending-decisions list (where it is stuck, the options, the recommendation with its reason, the completed work each option affects); a mismatch found between a document, a script, and the repository is corrected in the same change when small and certain, queued in that list when it needs a decision, or filed as an issue with `catalog/repository` when it is out of scope — never left in the conversation only.
- `.agents/knowledge/agent-authority.md`: a `## Pre-authorized actions` section — which outward actions a conversation's authorization covers and that the change record's proposal names the branch it pushes — and a `## Threat model` section recording that no unattended agent runs here, that the workflows reading issue and comment text are deterministic scripts without a model, and that the dev container holds no credential beyond the `gh` login the user forwards.
- `.claude/skills/change-workflow/SKILL.md ## 4`: the checkpoint rules — a checkpoint is a commit; a task's draft commits are squashed into its atomic commit before the task is ticked, with the non-interactive sequence (`git reset --soft` to the task's start, one commit under the commit rules); draft commits never leave the machine and every push names one ref; the per-commit hooks are the secret scan, so they stay seconds-fast — and a gotcha that handoff notes live in the scratch directory, never in the tree.
- `.devcontainer/devcontainer.json`: `PAGER`, `GIT_PAGER`, and `GH_PAGER` set to `cat` in `remoteEnv`, so unattended `gh` and `git` calls never wait on a pager; the pending `github-cli` feature addition in the working tree is included.
- `scripts/validate_harness.py`: a `workflows` check — every workflow declares top-level `permissions: {}` and every job its own permissions; every `uses:` is pinned to a commit SHA with a version comment; `pull_request_target` appears only in the workflows `github-checks.md` lists for it and those never check out the head; failures name the workflow file and the line to fix. `.agents/knowledge/github-checks.md` gains the row that names the workflows allowed to use `pull_request_target`, which the check reads.
- `.agents/knowledge/harness-maintenance.md`: a `## Layer calibration` table — the six layers, each rated with the variable that sets it (error cost: installed by others; maintenance horizon: indefinite; isolation: dev container, no secrets; autonomy: attended, H1; team size: one maintainer plus agents) — read by the periodic review, which gains the thinning question (which constraint has caught nothing since the last run).
- `.agents/knowledge/spec-workflow.md` and `openspec/schemas/skill-change/`: the artifact table gains a "Never holds" column and the two placement tests (invariant, information); the proposal template and instruction gain a `## Deliverable and pre-authorized actions` section naming what the change delivers and the branch and remote writes it pre-authorizes; the `skill-authoring` project skill and the pull request template are updated where the knowledge file's "Update this file when" list requires it.

## Skills touched

repository change.

## Installed behavior

Agents working in this repository are stopped by the framework before a force push, a hard reset, a merge, or a delete; continue past a blocked decision and hand over a pending list instead of stopping; land every harness mismatch they find in the change, the list, or an issue; squash a task's draft commits before ticking it and never push a draft; run `gh` and `git` without pager hangs; see workflow security rules fail as a check instead of a review comment; and read the harness's own layer calibration before changing its thickness. Proposals name their deliverable and pre-authorized actions, and the artifact table says where a fact goes.

## Impact

- Files: `.claude/settings.json`, `AGENTS.md`, `.agents/knowledge/agent-authority.md`, `.agents/knowledge/github-checks.md`, `.agents/knowledge/harness-maintenance.md`, `.agents/knowledge/spec-workflow.md`, `.claude/skills/change-workflow/SKILL.md`, `.devcontainer/devcontainer.json`, `scripts/validate_harness.py`, `openspec/schemas/skill-change/schema.yaml` and `templates/proposal.md`, `.github/PULL_REQUEST_TEMPLATE.md` if the proposal section is mirrored there, `.agents/skills/skill-authoring/SKILL.md` if it names the proposal sections.
- Mirrored pairs touched: `AGENTS.md` ↔ `ARCHITECTURE.md` (the new check and the calibration table are described there); `.github/PULL_REQUEST_TEMPLATE.md` ↔ change-workflow §6–7; the schema artifact table ↔ `spec-workflow.md` ↔ `skill-authoring`.
- `README.md`/`README.zh.md`: unchanged. `marketplace.json`: unchanged.

## Non-goals

- No change to any public skill; the methodology and the spec contract are the two sibling changes.
- No CI job that builds the dev container (the reference's pattern): the container holds no logic CI depends on beyond the tool pins the validator already checks.
- No hooks in `.claude/settings.json` that inject reminders after edits: the keep-in-sync pairs are mechanically checked already.
- No periodic-review run in this change; the calibration table is written, and the next quarterly run applies the thinning question.
- The `ask` list stays empty.

## Tracked work

No issue.
