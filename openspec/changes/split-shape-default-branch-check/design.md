## Context

See proposal.md for motivation. The companion repository change `split-shape-default-branch-check-harness` carries this repository's own `checks.yml`, `scripts/spec_changes.py` help text, and knowledge files.

**Current shape** (`origin/main` at 10e29c0):
- **Job assets.** `assets/github/job-spec-check.yml` passes `--shape {{REQUEST_SHAPE}}` on its pull-request branch (line 47) and runs `check --all` with no shape on its push branch (line 49). Its step comment (lines 27–32) says a push fails on any change outside `archive/`, "so the integration branch never holds an unarchived change". `assets/gitlab/ci-spec-jobs.yml` has the same two lines (59 and 61). Its job comment (lines 44–46) names only the merge-request rule.
- **Scripts.** `assets/spec_changes.py` (`cmd_check`, lines 811–824) and the bundled `scripts/spec_changes.py` (lines 749–760) already warn under `--all --shape split` and fail under the default, `combined`. The `--all` help of both (asset line 954, bundled line 940) says only "fail on any change outside archive/".
- **References.** `references/github.md` `## Ready-state rules` (lines 113–121) says `check --all` warns under split, and calls `combined` "the default, and what the asset ships", although the asset ships `{{REQUEST_SHAPE}}`. `references/gitlab.md` names `{{REQUEST_SHAPE}}` in its placeholder line (line 15) and states no push rule. Neither reference's `## Verification after installing` exercises the push branch.
- **SKILL.md.** `## Ready, then the freeze` (lines 82–84) says archiving inside the request means "the integration branch never holds an unarchived change", which is false under split. `## Commands and labels on a request` (lines 127–133) describes `--shape split` for the request check and ends "Combined projects leave the flag at its default and get no exception", which the new rule makes false: no `check --all` may rely on the default.
- **Spec.** The installation requirement's push clause, and the `--all` clause of both `Script:` requirements, state the combined rule alone. The archive requirement says nothing about what the integration branch holds; only SKILL.md does.
- **Intent.** Commit 4619653, which introduced `--shape`, says "`check --all` warns instead of failing under split … and both check assets pass it". The assets pass it on the request branch only.

**Binding constraints:**
- **`sdd/CONTEXT.md`.** Automation assets stay fork-safe by construction, and a script pins the tool flags it relies on with their verification date. Neither changes here: the push branch runs on the default branch, never on a request's head.
- **Management code (R1).** The asset and the bundled script are edited independently, and nothing requires them to stay identical.
- **Register.** `.agents/knowledge/harness-maintenance.md` pairs `job-spec-check.yml` with the `checks / spec` job of this repository's `checks.yml`, and the asset's command line with this repository's `scripts/spec_changes.py`. The companion change keeps that pair true.
- **Self-containment and size.** No path leaves the skill directory. The description does not change. SKILL.md stays well under 500 lines.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| ADDED Behavior: The default-branch check follows the recorded request shape | `assets/github/job-spec-check.yml`: the push branch runs `check --all --shape {{REQUEST_SHAPE}}`; the `{{REQUEST_SHAPE}}` comment moves above the `if`, so it covers both branches; the step comment states the push rule for each shape, and that under split the push warns whatever the task list says. `assets/gitlab/ci-spec-jobs.yml`: the same push line; the `spec:check` comment names the push rule, the split exception, and the same task-list sentence. `references/github.md` `## Ready-state rules`: "the default, and what the asset ships" is reworded so the job passes the recorded shape explicitly on both branches under either shape, and the paragraph says the split push warns whatever the task list says; `## Verification after installing`: one bullet running the push command under the recorded shape. `references/gitlab.md`: the placeholder line says `spec:check` uses `{{REQUEST_SHAPE}}` on both branches, with one sentence on the push rule for each shape; `## Verification after installing`: the same bullet. `SKILL.md` `## Ready, then the freeze`: the "never holds an unarchived change" clause holds for the combined shape, and a split project's integration branch holds approved records until their implementation requests archive them, so a warning about one is no reason to archive. `SKILL.md` `## Commands and labels on a request`: the closing combined sentence says combined projects pass `--shape combined` and get no exception, instead of leaving the flag at its default. | existing load sentences in `## Installing the automation` |
| MODIFIED Behavior: The automation is installed per platform from the skill's assets | Only the push clause of the gate parenthetical changes, and the files it lands in are those of the row above. No other file. | — |
| MODIFIED Script: spec_changes.py | `scripts/spec_changes.py`: the `--all` help text names both shapes. No logic change. | — |
| MODIFIED Script: assets/spec_changes.py | `assets/spec_changes.py`: the same help-text change, written for the asset. No logic change. | — |

## External impact

- **Companion change `split-shape-default-branch-check-harness`** (required by the register): `.github/workflows/checks.yml` pins `--shape combined` on its push step; this repository's `scripts/spec_changes.py` gets the asset's `--all` help; `.agents/knowledge/github-checks.md`, `spec-workflow.md`, and `harness-maintenance.md` quote the pinned push command. Proof: that change's verification plan and `just check`.
- **`spec-kit-workflow`: unaffected.** Its `check --all` (`assets/spec_kit_features.py` lines 707–715) checks only for a missing `spec.md` or `plan.md`, independent of shape, and a split specification request carries both. Proof: readback; `git diff --stat origin/main...HEAD -- skills/sdd/spec-kit-workflow` is empty.
- **`meta-spec-workflow`: out of scope.** Two passages keep the unqualified sentence: `references/contract-design.md` `## Archiving` ("the integration branch never holds an unarchived record"), and the deposited contract template `assets/spec-workflow.md` `## Archive executor` ("so <default branch> never holds an unarchived record"), which is written into target projects of either shape. Both are proposed for one follow-up issue. Proof: `git diff --stat origin/main...HEAD -- skills/meta/meta-spec-workflow` is empty.
- **No change to** the description, the symlink, `marketplace.json`, the `sdd` README pair, or `skills/sdd/CONTEXT.md`. Proof: `just validate`.

## Decisions

- **Under split the push check warns; under combined it fails** (ADDED requirement; maintainer decision).
  - The script, `references/github.md`, and commit 4619653 all state this rule. Only the job assets and the spec sentence disagree with it.
  - Rejected: always fail, by deleting the scripts' `--all` split branch and changing the reference. A split project's default branch would be red between every specification request and its implementation request, which teaches maintainers to ignore red.
- **Fix it in the job assets with `--shape {{REQUEST_SHAPE}}` on the push line** (ADDED requirement; maintainer decision).
  - The install already resolves this placeholder from the contract for the request branch, and the leftover-placeholder check in each reference catches it unresolved on either line.
  - Rejected: a split default in the script, which would silently weaken every combined install that leaves the flag out.
  - Rejected: the script reading the shape from the project's contract, a new interface to a file whose location and format the skill does not own.
  - Rejected: a separate placeholder for the push branch, which could be resolved to a different shape than the request branch.
- **Under split, keep warning even for a change with a ticked task** (ADDED requirement, scenario "Split shape, a record with a ticked task"; maintainer decision).
  - The request check already fails a request that completed a task without archiving, so such a record reaches the default branch only if that check was bypassed.
  - Rejected for now: failing the push check when a task is ticked. It depends on whether a split record may be implemented across several requests (`meta-spec-workflow`'s contract design has "whose last implementation request closes"), where a partly implemented record on the default branch is legitimate. Proposed as a follow-up.
- **No script logic changes; help text only** (both `Script:` requirements).
  - Both scripts already behave as the rule says. The asset and the bundled script get their help text separately, under R1.
  - The bundled script's `Script:` clause also gains the request-side split admission it already implements, because the clause is rewritten here anyway and would otherwise state half the shape rule.
- **Qualify SKILL.md for split** (ADDED requirement; maintainer decision).
  - Loaded on a split project, the unqualified sentence tells the agent the default branch must be clean, which invites archiving a record whose implementation has not started. The same edit reaches `## Commands and labels on a request`, whose "leave the flag at its default" would contradict the rule.
  - Rejected: leaving it, since `## Ready, then the freeze` loads on every archive question.
  - Rejected: also rewording `meta-spec-workflow`'s `contract-design.md` and `assets/spec-workflow.md` here. That is another skill, with its own deposited contract text, and belongs in a follow-up.
- **One ADDED requirement for the rule; MODIFIED only where an existing clause contradicts it** (every requirement; the maintainer's preference for ADDED, to be confirmed).
  - The ADDED `Behavior: The default-branch check follows the recorded request shape` holds the rule, the SKILL.md qualification, and all five new scenarios, so no other change has to re-copy them.
  - Three existing clauses state the combined rule alone and would contradict the new requirement: the push clause of the installation requirement, and the `--all` clause of each `Script:` requirement. They are MODIFIED, each with only that clause edited. The `Script:` blocks also carry their new scenarios, because a script has one `Script:` requirement and its scenarios cannot stand alone.
  - The archive requirement is not modified: it states nothing about what the integration branch holds, so nothing in it contradicts the rule.
  - Order: at archive the pinned CLI (OpenSpec 1.12.0, checked 2026-09-30 by a scratch archive of this delta, which then passed strict validation) appends the ADDED block after `Script: assets/spec_changes.py`, out of the schema's Trigger, Behavior, Handoff, Script order. Main specs already break that order (`meta/meta-spec-workflow` has a Behavior after its Handoff; `sdd/spec-driven-development` has eleven Behaviors after its first Handoff), the strict validator does not check it, and the kind prefix in each name still lets tests derive from it. #96+#100's ADDED blocks land the same way.
  - Rejected: MODIFIED blocks only, with the rule's scenarios inside the installation and archive requirements (the previous draft). It widens the blocks another change must re-copy, and its reason, that no main spec breaks the order, was false.
  - Rejected: reordering the main spec by hand after the archive command. It edits the tool's output, and every later archive appends again.
- **Separate from #96, #100, and #101** (maintainer decision). This change is the smallest of the sdd changes and lands first. #96+#100 adds requirements rather than modifying these blocks. #101 modifies `Script: assets/spec_changes.py` and rebases on this change.

## Risks / Trade-offs

- **[Another sdd change archives over this one's blocks, or this one over another's]** → #99, #96+#100, and #101 all touch `openspec/specs/sdd/*`; the planned merge order is #99, then #96+#100, then #101. A MODIFIED block replaces the main block whole at archive. So after a preceding sdd pull request merges, this branch rebases on `main` and re-copies every MODIFIED requirement block from the then-current main spec before its archive commit, re-applying only its own clause edits and scenarios. No archive then silently reverts another change's clause. The pull request states this, and later sdd changes follow the same rule against this one. The ADDED requirement needs no re-copy.
- **[Projects that installed the automation earlier keep the failing push line]** → Installed copies are never updated in place. The pull request's description gives the one-line change (`check --all --shape split`) for split projects. Combined projects need nothing.
- **[Under split, the push check never fails an unarchived change]** → Accepted. The request check stays the guard, and the warning names every record still waiting. A stricter rule is a follow-up.
- **[The resolved shape on the two lines of one job disagrees]** → One placeholder, one comment above both branches, and the verification bullet that runs both branches under the recorded shape.
- **[`meta-spec-workflow` keeps the unqualified sentence]** → A builder-deposited contract may still say the default branch never holds an unarchived record, through `assets/spec-workflow.md` `## Archive executor`, and `references/contract-design.md` `## Archiving` still teaches it. The follow-up issue names both files and sections.
- **[The ADDED block sits out of the schema's order in the main spec]** → Accepted, as in the two main specs that already do so; the name's kind prefix, not its position, identifies it.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Solver tier:** Sonnet-class, the least capable tier past changes certified this skill at.

**Observation:**
- Loading: the framework's native skill-load history, otherwise the neutral `SKILLS_LOADED:` self-report.
- Outcomes: the transcript plus the fixture's diff against its initial commit.

**Isolation:**
- One fresh clean-context subagent per case.
- Each fixture is a throwaway git project under the session scratch directory.
- The candidate skill is copied into the user skills directory for the run and removed afterwards.
- One attempt, and up to three when the observation is invalid.
- An independent clean-context grader on a capable model grades anonymized outputs.

No description changes, so there are no trigger cases.

Each rubric item scores 1. Critical items are marked (C), and any critical failure fails the case.

**Fixtures:**
- `gh-openspec-split`: a GitHub repository with a `checks` workflow and gate, a `labels.json`, an `openspec/` directory, and `.agents/knowledge/spec-workflow.md` recording the split request shape.
- `gh-openspec-combined`: the same, with the combined shape recorded.
- `gl-openspec-split`: a GitLab project with a `.gitlab-ci.yml`, an `openspec/` directory, and the split shape recorded.
- `gh-openspec-split-installed`: `gh-openspec-split` with the automation installed from the candidate assets, and `openspec/changes/add-export/` merged on `main` with no ticked task.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| Default-branch check: Split shape, push after a specification request; Installation: GitHub install | O1 in `gh-openspec-split`: "Install the OpenSpec request automation in this repository." | every `check` line of the produced spec job carries `--shape split`, the push branch's `check --all` included (C); `grep -rn '{{[A-Z]' .github/` returns nothing (C); the produced push command, run against a scratch copy holding `add-export` with no ticked task, exits 0 with a warning naming it (C); the agent says a push to the default branch warns about an approved, unarchived record instead of failing; the check job joins the gate's `needs:` and nothing pushes | all (C) and ≥ 4/5 | Sonnet-class | transcript + diff + command run | as above |
| Default-branch check: Split shape on GitLab; Installation: GitLab install | O2 in `gl-openspec-split`: "Install the OpenSpec merge request automation in this GitLab project." | both branches of `spec:check` carry `--shape split` (C); `grep -n '{{' .gitlab-ci.yml <fragment>` returns nothing (C); the push branch's command, run with `CI_MERGE_REQUEST_IID` unset against a scratch copy holding an unticked change, exits 0 with a warning naming it (C); the agent states that labels take effect on the next pipeline and that manual jobs replace comment commands | all (C) and ≥ 3/4 | same | transcript + diff + command run | same |
| Default-branch check: Combined shape, push | O3 in `gh-openspec-combined`: "Install the OpenSpec request automation in this repository, and tell me what happens on main if a change is ever merged without being archived." | the push branch carries `--shape combined` (C); the produced push command, run against a scratch copy holding an unticked change, exits 1 naming it (C); the agent says the check on the default branch fails on such a change (C) | all (C) | same | transcript + diff + command run | same |
| Default-branch check: Warning on a split project's default branch | O4 in `gh-openspec-split-installed`: "spec / check on main warns that add-export is unarchived since we merged its spec PR. Should I archive it now to clear the warning?" | the agent says the warning is expected because the record awaits its implementation request (C); it archives nothing, and neither runs nor recommends the archive command now (C); it says the implementation request archives the record once its deliberation closes | all (C) and ≥ 2/3 | same | transcript + diff | same |

**Readback cases.** A clean-context subagent reads the finished skill files. For each scenario it quotes the passage that produces the scenario's THEN, and says whether that passage is present, precise, and unconditional. A scenario with no passage is a critical failure. Threshold: every scenario has a passage.
- Default-branch check: Split shape, a record with a ticked task (the job comments and `references/github.md` `## Ready-state rules` state that the push check warns under split whatever the task list says).
- Regression of the edited `## Ready, then the freeze`: Archive: After the deliberation, Asked to archive before the deliberation, and Open task; Installation: Ready with an unarchived change.
- Regression of the edited `## Commands and labels on a request` and `## Ready-state rules`: Installation: Combined shape, nothing implemented, and Split shape, the specification request. The readback also confirms that no passage of the skill still says a combined install leaves `--shape` at its default or that the asset ships `combined`.

**Script and tool harnesses.** These are untracked scratch repositories under the session scratch directory, run with `bash`. The scratch repository runs `openspec init` with the pinned CLI and commits, on `main`, one change `add-thing` whose `tasks.md` has no ticked task.
- **`assets/spec_changes.py` and the bundled `scripts/spec_changes.py`,** each:
  - `check --all` exits 1 with `unarchived change add-thing: the integration branch holds only archived changes.` (Integration branch under each shape).
  - `check --all --shape split` exits 0 with `warning: unarchived change add-thing on the integration branch (expected under split).` (Integration branch under each shape).
  - After one task of `add-thing` is ticked and committed, `check --all --shape split` still exits 0 with the same warning (Default-branch check: Split shape, a record with a ticked task, at script level).
  - `check --help` exits 0, and its `--all` line names both shapes (Integration branch under each shape).
  - `--help` exits 0 and names the five subcommands (asset) or the seven (bundled) (Help). `--bogus` exits 2 and names the option (Bad arguments).
  - An identical second `check --all --shape split` gives identical output, and `git status --porcelain` stays empty (Repeated run).
  - Bundled only: on a branch that adds a change with no ticked task, `check --base main --head HEAD --shape split` exits 0 with a warning naming it; after one task is ticked and committed, it exits 1 naming it (Specification request under split).
- **Job assets:**
  - Each asset is resolved into two scratch copies, `split` and `combined`, with every placeholder filled. Each copy parses with PyYAML; the GitHub copy also runs through `actionlint` when it is available, and its absence is recorded.
  - `grep -n '{{[A-Z]'` over the resolved copies returns nothing.
  - Each copy's push-branch body runs under `bash -e` against the scratch repository (GitHub: `EVENT=push`; GitLab: `CI_MERGE_REQUEST_IID` empty): `split` exits 0 with the warning, `combined` exits 1 naming `add-thing` (Split shape, push after a specification request; Split shape on GitLab; Combined shape, push).
  - After one task of `add-thing` is ticked and committed, each `split` copy's push-branch body runs again the same way and still exits 0 with the warning naming `add-thing` (Split shape, a record with a ticked task).
  - `grep -rn 'check --all' skills/sdd/openspec-workflow/assets` shows `--shape` on every hit.
- **Repository gates:** `just check-skill skills/sdd/openspec-workflow`, `just validate`, `just spec-validate`, `just lint`, and `just check`.

Skipped: none planned. If clean-context subagents are unavailable when the tests run, O1–O4 and the readback cases are recorded as skipped with that reason, as the subagent gate requires.
