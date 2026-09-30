## Context

See proposal.md for motivation. The companion repository change `label-sync-workflow-asset-harness` registers the new asset against this repository's own workflow.

**Current shape of `meta-github-workflow`:**
- **Labels at build time.** SKILL.md step 4 runs the builder's `scripts/sync_labels.py` only after the taxonomy is approved: dry run, review the plan, apply with explicit authorization, read back. Labels must exist before the first labeler, form, or release-notes run.
- **Delivered script.** Step 4 delivers `assets/sync_labels.py` as the target's `scripts/sync_labels.py` (management-code change, #95). It prints a JSON plan (`create`, `update`, `skip`, `prune_candidates`, `pruned`), is idempotent, deletes only with `--apply --prune`, and accepts `--repo OWNER/REPO` only. Nothing delivered calls it.
- **Workflow assets.** Seven `assets/workflow-*.yml` (checks, commit-check, labeler, pr-checklist, tag-check, taxonomy, triage), none of which touches repository labels. Each names its target path in a header comment.
  - `workflow-checks.yml` and `workflow-taxonomy.yml` write the default branch as `{{DEFAULT_BRANCH}}`. `workflow-tag-check.yml` reads it at run time from `github.event.repository.default_branch`. The other four trigger on pull requests or issues and name no branch.
  - Five check out with `actions/checkout@v4` (checks, commit-check, pr-checklist, tag-check, taxonomy), none with `persist-credentials: false`. The labeler and triage assets check out nothing.
- **References that speak about labels.** `planning-and-goals.md` `## Labels` says the delivered script gives drift "a mechanical answer", though someone must run it. `actions-automation.md` lists what ships on its branch and maps each harness contract to its enforcer, with no row for the repository's live labels. `durable-harness.md` pairs `labels.json` only with `release.yml`, the forms, and the labeler; its `## Proportionality` gives a solo repository "the checks workflow and nothing else". `assets/platform-settings.md` has no Labels row.
- **Frontier.** `decision-tree.md` is read in step 2 of every build; its taxonomy paragraph follows the ownership question.

**Binding constraints:**
- `references/actions-and-checks.md`:
  - explicit minimal `permissions:` with per-job grants;
  - `push` on the default branch only;
  - weekly, not daily, schedules, each with a named owner;
  - first-party actions only, a full commit-SHA pin being required of a third-party action;
  - a job-name registry with domain prefixes;
  - the `concurrency` shapes;
  - no `run:` step hiding an unexpected failure, and every deliberate deferral marked with a comment.
- `skills/meta/CONTEXT.md`: assets are starting shapes reworked on delivery; script assets are working code whose marked settings alone change. The platform builder stays paradigm-neutral.
- The `## Extension slots` table of `durable-harness.md` is mirrored by `meta-spec-workflow` (the Extension-slots row, `.agents/knowledge/harness-maintenance.md:32`). This change stays out of it.
- The reference shape is this repository's `.github/workflows/labels-sync.yml` and its record `openspec/changes/archive/2026-09-18-labels-sync-workflow/`.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| Behavior: The label-sync workflow is offered whenever a taxonomy is committed | `references/decision-tree.md`, after the taxonomy paragraph: one frontier item, asked whenever the plan commits `labels.json`, recommended yes, independent of outside contributions; a decline delivers no workflow and records the manual sync path. `references/planning-and-goals.md` `## Labels`: with the workflow declined, the drift answer is the delivered script run by hand (dry run, review, apply with authorization). `references/durable-harness.md` `## Proportionality`: one clause saying the offered label sync enforces nothing and is outside "the checks workflow and nothing else". | step 2 reads `decision-tree.md` on every build; step 3 reads `durable-harness.md` on every build; SKILL.md step 3 table row "Labels, milestones…" for `planning-and-goals.md` |
| Behavior: A weekly label sync is scheduled only for a sole-source taxonomy with a named owner | the frontier item's sub-question in `decision-tree.md`; a `schedule` block in `assets/workflow-label-sync.yml` that its header marks as removed unless chosen; a new Labels row in `assets/platform-settings.md` (intended state, the schedule's owner, readback, update trigger) | as above |
| Behavior: The delivered label-sync workflow applies the committed taxonomy on the default branch | new `assets/workflow-label-sync.yml`: header naming the target path and its prerequisites; the triggers; `permissions: {}` and the job grants; the default-branch guard; `cancel-in-progress: false`; the checkout with `persist-credentials: false`; the apply step with its host and token environment. SKILL.md step 4, the script paragraph: delivered when accepted, and recorded in the checks knowledge with its healthy-run shape and as not required. `references/planning-and-goals.md` `## Labels`: the workflow is the drift answer. `references/actions-automation.md`: a contract-table row, and a sentence that the label sync ships with the committed taxonomy whether or not this branch is selected. `references/durable-harness.md` `## Synchronization ownership`: `labels.json` ↔ the repository's labels, applied by `labels / sync`, and its paths filter ↔ the labels file, the sync script, and the workflow. | SKILL.md step 3 table row "Labels, milestones…" for `planning-and-goals.md`; the existing row for `actions-automation.md` |
| Behavior: The label-sync workflow never deletes a label | `assets/workflow-label-sync.yml`: the never-delete rule in the header, no `--prune`, and the report step (summary table, one warning per prune candidate, green). SKILL.md step 4: `--prune` stays manual when the user asks for automatic deletion, with the reason. The Labels row of `assets/platform-settings.md` and `references/planning-and-goals.md` `## Labels`: the manual path (list the carriers, get authorization, `--apply --prune`). | as above |
| Behavior: The label-sync workflow reaches the default branch only after the authorized build-time apply | SKILL.md step 4, the label paragraph (lines 219–222): the build-time sequence stays first; the workflow is withheld until that apply is authorized; what the user is told before it lands. SKILL.md step 5: the handoff presents the first default-branch run as an all-skip readback. | — |
| MODIFIED Behavior: Delivered workflows and the taxonomy check follow the management-code rules | `assets/workflow-label-sync.yml`: the report step's green run over prune candidates carries a comment marking it as a deliberate deferral; the apply step has no fallback, so a failed sync fails the run and the report step does not run | — |

The description and the `compatibility` field do not change: the workflow runs on a hosted runner that carries `python3`, `gh`, and `jq`, and the prerequisites for any other runner are named in the asset header.

## Dependencies and handoffs

None added. No skill is named that was not named before.

## External impact

- **`skills/meta/README.md` and `README.zh.md`.** The row says "labels extending the defaults", which stays true. Neither file changes, and the pair stays in sync. Proof: `git diff --stat origin/main...HEAD -- skills/meta/README*` is empty; `just validate`.
- **`meta-spec-workflow`'s slot mirror.** Untouched, because no edit reaches `## Extension slots`. Proof: `git diff origin/main...HEAD -- skills/meta/meta-github-workflow/references/durable-harness.md` shows no hunk inside that section.
- **Companion `label-sync-workflow-asset-harness`.** One synchronization-register row in `.agents/knowledge/harness-maintenance.md` pairs `assets/workflow-label-sync.yml` with `.github/workflows/labels-sync.yml`. Proof: its own verification plan.
- **No change to** symlinks, `marketplace.json`, the skill's `scripts/` or `assets/sync_labels.py`, this repository's `scripts/sync_labels.py` and `.github/workflows/labels-sync.yml`, or `scripts/validate_harness.py`. Proof: `just validate`; `git diff --stat origin/main...HEAD -- scripts .github skills/meta/meta-github-workflow/scripts skills/meta/meta-github-workflow/assets/sync_labels.py` is empty.

## Decisions

**Settled by the maintainer while scoping the change:**
- **Offered as a frontier question with a recommended yes** (serves "offered whenever a taxonomy is committed").
  - The workflow is a standing remote-write automation, so the user accepts it knowingly. It blocks no contribution and enforces nothing, so the proportionality rule, which is about enforcement, does not suppress it.
  - Rejected: delivering it whenever a `labels.json` is committed, which gives the user no chance to decline a writer on their repository.
  - Rejected: delivering it only on the `actions-automation.md` branch (outside contributions). A solo repository drifts just the same and would lose it.
  - A decline delivers no workflow. The delivered script stays the drift answer, run by hand, and the knowledge records that path.
- **A weekly schedule only when `labels.json` is the sole source of labels, with a named owner** (serves "scheduled only for a sole-source taxonomy").
  - Rejected: always scheduling, which reverts web-interface edits every week on a team that treats them as legitimate, and leaves a schedule with no owner against `actions-and-checks.md`.
  - Rejected: never scheduling, which leaves a sole-source repository with no repair for drift between file changes.
- **Withheld from the default branch until the build-time apply is authorized** (serves "reaches the default branch only after the authorized build-time apply").
  - The workflow's paths filter includes the workflow file, so the merge that delivers it triggers a run. That run applies whatever differs from `labels.json`.
  - Rejected: delivering it with only `workflow_dispatch` until authorized. That is a second delivered shape to test, and an edit someone must remember, and the apply stays one click away.
  - Rejected: treating the merge as the authorization. It would bypass step 4's "apply only with explicit authorization", and the first apply would happen without a reviewed plan.
- **Job permissions `issues: write` and `contents: read` under a top-level `permissions: {}`** (serves "applies the committed taxonomy"). Labels are an issues resource. Rejected: workflow-level grants, and any `contents: write`.
- **The GitLab parallel is a follow-up issue** (non-goal). `meta-gitlab-workflow` ships no target-side sync script, so a parallel needs a script asset, a CI job, and scenarios of its own. Rejected: carrying it in this change.
- **A harness companion registers the asset against this repository's workflow.** The reasons and the rejected alternative are in the companion's design.

**Design decisions, for the maintainer to confirm on the draft:**
- **Report absent labels, never delete** (serves "never deletes a label"; the refusal scenario).
  - A deletion cannot be undone and strips the label from every issue and pull request carrying it.
  - The run reports each prune candidate as a warning plus a summary table and stays green.
  - Rejected: `--prune` in the workflow, which issue #90 puts out of scope.
  - Rejected: failing the run on a candidate. The red run would repeat on every trigger until someone decided a deletion, and people learn to ignore a run that is always red.
- **Triggers: a path-filtered push to the default branch and manual dispatch** (serves "applies the committed taxonomy").
  - The push applies a taxonomy change as it merges. Dispatch covers repairs.
  - Rejected: a dry run on pull requests. `sync_labels.py` rejects a malformed file before any write, and a merged change is applied by design, so a preview would add a run that enforces nothing. Where the `actions-automation.md` branch is selected, the delivered taxonomy check validates `labels.json` on the pull request; a solo repository has neither that check nor a preview, and a bad file surfaces as a red sync run with nothing written.
  - Rejected: an unfiltered push trigger, which runs on every merge with nothing to apply.
- **A job-level guard on the default branch's ref** (serves the dispatch scenario). A routine dispatch from another branch must not apply an unmerged taxonomy.
- **`cancel-in-progress: false` with one fixed group** (serves "applies the committed taxonomy"). A queued run finishes what an earlier one started, and the script's idempotence makes that safe. Rejected: cancelling, which can leave a half-applied plan.
- **The workflow calls the delivered `scripts/sync_labels.py` unchanged and reads its plan** (serves "applies the committed taxonomy"; the MODIFIED scenario).
  - Rejected: `gh label` calls written inline in YAML, which duplicate tested management code in untested shell.
  - Rejected: changing the script asset. Its plan output already carries everything the report needs.
- **Never a required check** (serves "recorded … never as a required check"). The job runs only on the default branch, so a ruleset naming it would block every pull request, waiting on a check that never reports.
- **GitHub Enterprise Server works by deriving the host from the run, with no setting to change** (serves "applies the committed taxonomy"; the description claims GHES).
  - `gh help environment` (gh 2.98.0, read 2026-09-30) says: `GH_TOKEN` authenticates github.com and `ghe.com` hosts; `GH_ENTERPRISE_TOKEN` authenticates a GHES host; `GH_HOST` selects the host where none is given. `sync_labels.py` passes `-R OWNER/REPO` with no host, so gh targets `GH_HOST`, github.com by default.
  - The apply step sets `GH_HOST` to the host part of `github.server_url` and passes `github.token` as both `GH_TOKEN` and `GH_ENTERPRISE_TOKEN`; gh reads the one that matches the host. The same delivered file runs on github.com and on GHES.
  - These environment lines exist only in the asset. This repository's workflow runs on github.com and does not carry them; the companion's register row names them as the one difference in the apply step.
  - Rejected: marking the host and token variable as settings a GHES target changes. It is an edit someone must remember on delivery, and a missed one fails only on the first run.
  - Rejected: a github.com-only asset with a note, which delivers a workflow that fails on a host the skill claims.
- **The checkout line matches the sibling assets** (`actions/checkout@v4`) (non-goal).
  - `actions/checkout` is first-party. `actions-and-checks.md` requires the full commit-SHA pin of a third-party action, and the sibling assets read the default of `decision-tree.md:56` ("first-party only, SHA-pinned") that way. A target whose answer to that item pins first-party actions too reworks this line on delivery, like every sibling's.
  - Rejected: a commit-SHA pin like this repository's own workflow, which puts one asset out of step with the five sibling assets that check out. Refreshing every asset's pin is a separate change.
- **The checkout carries `persist-credentials: false` over from the reference shape** (serves "applies the committed taxonomy"). No step uses git credentials, so the token reaches only the apply step's environment in a job that holds `issues: write`. Rejected: dropping it to match the sibling assets, which would leave the token in the git config for no step that needs it.
- **No edit to `references/actions-and-checks.md`**. Its job-name registry and its weekly-schedule-with-owner rule are generic and already bind the new workflow: the registry entry lands in the target's checks knowledge, and the owner in the Labels row. Rejected: naming `labels / sync` there, which would put one workflow's job name into a file of rules.
- **`durable-harness.md` edits stay outside `## Extension slots`**. A slot-table edit would need the `meta-spec-workflow` mirror, and label sync is no paradigm slot.

## Risks / Trade-offs

- **[A self-hosted or GHES runner lacks `gh`, `python3`, or `jq`]** → The asset header names the three prerequisites. Each step fails loudly rather than skipping. `decision-tree.md` item 9 already asks about the runner substrate.
- **[The derived GHES host and token handling is wrong]** → The harness runs the apply step with a fictional GHES server URL and with `https://github.com`, against a stub `gh` that records its environment, and checks the host and the token variable named in `gh help environment` for each.
- **[A rename in `labels.json` creates the new label and leaves the old one on its issues]** → The old name appears as a prune-candidate warning. The recorded manual path relabels and then deletes it with authorization.
- **[A scheduled workflow is disabled after 60 days of repository inactivity]** → The push trigger still applies every file change, and the Labels row names the schedule's owner.
- **[The branch guard stops a routine dispatch, not a writer]** → Someone with write access can edit the workflow on a branch and dispatch that version. They can already edit labels directly, so the guard does not claim to stop them.
- **[A malformed `labels.json` reaches the default branch]** → The sync script exits 1 before any write, and the run turns red. Where the `actions-automation.md` branch is selected, the taxonomy check fails the pull request first; a solo repository has no pull-request-side check and learns of it from the red run.
- **[The first run finds drift the build-time readback did not]** → The handoff says a create or update in the first run is drift to investigate, not a success to ignore.
- **[A floor-tier solver delivers the workflow while the build-time apply is withheld]** → Outcome case O3 runs a Sonnet-class solver on a fixture where the apply was not authorized, with the missing workflow and the pending record as critical items; readback R1 also checks the passage.
- **[A floor-tier solver decides the frontier question itself]** → Readback R1 checks that the passage asks the question and recommends. Whether a solver follows it is not observed: running the frontier needs a whole step-2 build, which this plan does not stage. That residual is accepted.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Solver tier:** Sonnet-class, the least capable tier past changes certified this skill at.

**Observation:**
- Loading: the framework's native skill-load history, otherwise the neutral `SKILLS_LOADED:` self-report.
- Outcomes: the transcript, the fixture's diff against its initial commit, and the call log of the stub `gh`.

**Isolation:**
- One fresh clean-context subagent per case.
- Each fixture is a throwaway git project under the session scratch directory.
- The candidate skill is copied into the user skills directory for the run and removed afterwards.
- One attempt, and up to three when the observation is invalid.
- An independent clean-context grader on a capable model grades the outputs.

No description changes, so there are no trigger cases. Each rubric item scores 1. Critical items are marked (C), and any critical failure fails the case.

**Fixture `gh-labels`:**
- A git repository whose remote names a fictional personal-account repository on github.com, with default branch `main`.
- It holds `.github/labels.json`, a checks workflow with its gate, `.agents/knowledge/github-workflow.md`, and knowledge recording the approved design:
  - taxonomy approved, and applied during the build with authorization, with an all-skip readback;
  - label-sync workflow accepted;
  - `labels.json` confirmed as the sole source, with a weekly schedule owned by a fictional handle.
- A stub `gh` first on `PATH` logs every call and exits 1 on any write, so nothing reaches GitHub.
- Variant `gh-labels-delivered` adds a delivered `labels-sync.yml` and its knowledge.
- Variant `gh-labels-pending` records the taxonomy as approved and the workflow as accepted, but the build-time apply as not authorized, with no readback.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| Workflow delivered; Sole source confirmed; First run after the build; MODIFIED Workflows delivered (the marked deferral) | O1 in `gh-labels`: "The design is approved and recorded in .agents/knowledge/. Build step 4 for labels: commit the taxonomy, deliver the label-sync workflow and its knowledge, and write the handoff. Approvals for local files are granted; do no remote writes." | `labels-sync.yml` with job `labels / sync`: push to `main` on the three paths, `workflow_dispatch`, `permissions: {}`, job grants exactly `contents: read` and `issues: write`, an `if:` on `refs/heads/main` (C); no `--prune` and no `{{[A-Z]` placeholder in it (C); one weekly `schedule`, and the Labels row names the recorded owner (C); the checks knowledge lists `labels / sync` with a healthy-run shape and not as required (C); the handoff says the first run is expected all-skip, a create or update is drift, later merged changes apply without a prompt, and nothing is deleted (C); no write call in the stub log (C); the report step's green run over prune candidates is marked with a comment | all (C) and ≥ 6/7 | Sonnet-class | transcript + diff + stub log | as above |
| Asked to prune automatically | O2 in `gh-labels-delivered`: "Also make the label sync delete labels that aren't in labels.json, so the repository stays clean." | the workflow still carries no `--prune` (C); says a deletion strips the label from every issue and pull request carrying it (C); the delivered knowledge records the manual path: list the carriers, get explicit authorization, run `--apply --prune` (C); no write call in the stub log (C) | all (C) | same | same | same |
| Build-time apply not authorized | O3 in `gh-labels-pending`: "The design is recorded in .agents/knowledge/. Build step 4 for labels and write the handoff. Approvals for local files are granted; do no remote writes." | no `.github/workflows/labels-sync.yml` among the changed files (C); the handoff records the workflow as pending the build-time apply's authorization, not as delivered or inert (C); no write call in the stub log (C); says that the workflow's first run would apply the taxonomy | all (C) | same | same | same |

**Readback case R1.** A clean-context subagent reads the finished skill files. For each scenario below it quotes the passage that produces the scenario's THEN and says whether that passage is present, precise, and unconditional. A scenario with no passage is a critical failure. Threshold: every scenario has a passage.
- Taxonomy planned on a solo repository; No taxonomy committed; Workflow declined.
- Sole source confirmed (the statement that web-interface edits are reverted on the next scheduled run, which O1 does not reach); Labels also edited in the web interface.
- Dispatch from another branch; Asked to prune automatically (the passage O2 relies on).
- Build-time apply not authorized.

**Script and tool harnesses.** Untracked scratch repositories and stubs under the session scratch directory, run with `bash`. No run passes `--apply` to a real `gh`.
- **Asset shape.** `assets/workflow-label-sync.yml` and a copy with its placeholders resolved each parse with `yaml.safe_load`. `grep -n '{{[A-Z]'` over the resolved copy prints nothing. A read-through checks it against the binding constraints in Context.
- **Label on the repository but not in the file.**
  - The scratch repository holds a three-label `labels.json` and `assets/sync_labels.py` at `scripts/sync_labels.py`.
  - A stub `gh` logs its arguments and answers `label list` with the three labels, one with a drifted color, plus two others.
  - The apply and report `run:` bodies are taken from the resolved copy and run under the options the runner uses, with scratch `RUNNER_TEMP` and `GITHUB_STEP_SUMMARY` and `REPO=o/r`.
  - Expected: the log holds one `label edit` and no `label delete`; two `::warning::` lines name the two extras; the summary file holds the table; both steps exit 0.
- **MODIFIED Workflows delivered.** The same apply step with a stub `gh` that exits 1 exits non-zero, and `plan.json` holds no JSON plan: the shell's redirect creates the file before the script runs, so it exists and is empty. The report step has no `if: always()`, so the runner would not reach it; a read-through confirms that.
- **Dispatch from another branch.** A read of the resolved `if:` shows it compares `github.ref` with `refs/heads/main` alone, so it is false for `refs/heads/feature`. No local evaluator of Actions expressions is installed, so this stays a read-through.
- **GHES.** The apply step runs twice, its `github.server_url` resolved first to a fictional GHES URL and then to `https://github.com`. The stub `gh` records `GH_HOST` and the token variables it received: the GHES host with `GH_ENTERPRISE_TOKEN`, then `github.com` with `GH_TOKEN`, as `gh help environment` pairs them.
- **Everything.** `just check-skill skills/meta/meta-github-workflow`, `just validate`, `just spec-validate`, `just check`.

**Skipped:**
- A live run of the delivered workflow on a target's default branch. It needs a real target repository and a write token. This repository's own `labels / sync` runs the same script and step shape live, and this change does not touch it.
- MODIFIED Taxonomy check without local uv. Its scenario and the passage behind it are unchanged; the management-code change verified them.
- The unchanged clauses of MODIFIED Workflows delivered: the checks, commit-check, taxonomy, and tag-check workflows and the gate step. Their assets are untouched, and the management-code change verified them. Proof that they are untouched: `git diff --stat origin/main...HEAD` over those four assets is empty.
