## Context

See proposal.md. The skill lives in `core`. Core skills are installed globally and load into every session, so the description must stay tightly scoped, and the change may name no skill outside `core` (`skills/core/CONTEXT.md`). The current description is a folded scalar of 483 characters (`SKILL.md` lines 3–10). The misbehaviour branch sits fourth in its Use-when list, and nothing in it names a skill that "never gets picked up" or separates a skill problem from a location problem.

Size limits, as the checks read the folded value:
- `scripts/check_skill.py` (`DESC_MAX = 1024`, `DESC_WARN = 900`) is what `just check-skill` and `just validate` enforce: an error above 1024, a warning above 900.
- The skill's own `scripts/lint_skill.py` (`DESCRIPTION_WARN_MILD = 600`) warns above 600. The skill must pass its own linter, so 600 is the budget.

The domain `openspec/specs/core/great-skill-writing/spec.md` exists. Its `Trigger: description` block has three scenarios. The "Misbehaving skill, indirect phrasing" scenario was removed from it (cb49b76) before `lint-skill-help` archived, because it failed at the Sonnet tier.

A second copy of the old description is installed globally as `great-skill-writer` in the user-level skills directory (`~/.agents/skills/`, which `~/.claude/skills` links to). It is not a link into the repository, so an edit here does not reach it, and the agent's sandbox cannot write there. In the #82 run, every session listed both copies.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| Trigger: description | `skills/core/great-skill-writing/SKILL.md` frontmatter `description`; nothing else in the skill changes | — |

## Description

What the description must contain:
- **Capability**: create, review, and repair Agent Skills. The capability sentence stays in the third person.
- **Must load**:
  - a skill or agent instruction package that never gets picked up or never triggers. It must load in the indirect phrasing that never says "skill" ("the instruction package I gave my agent … never gets picked up") and in the direct phrasing that names a location ("… its SKILL.md is under .agents/skills …"). A plausible location cause in the request must not pull the load decision away from the skill.
  - every trigger the current description already carries: the request names an Agent Skill, SKILL.md, or agent instruction package, or such a file is in the material; drafting reusable instructions or a prompt package for an agent; a skill that fires on the wrong task or gives inconsistent output; skill versus baseline agent docs.
- **Must not load**: application code, documentation written for people ("write a README …"), human skills ("which skills should a junior engineer build first?"), and something other than a skill that never triggers (a git hook, a CI trigger, a cron job).
- **Budget**: at most 600 characters as the linters measure it. No synonym list for "does not load"; one phrasing per branch.

## External impact

None. The `core` README pair rows describe the capability ("trigger-accurate descriptions, progressive disclosure, …"), not the trigger wording, and no sync rule ties them to the description, so both stay unchanged. No file is added, moved, or removed: the `.agents/skills/` symlink, `marketplace.json`, `skills/core/CONTEXT.md`, `scripts/validate_harness.py` (it mirrors no description), and the knowledge base are untouched. Proof: `just validate`, and `git status --short` listing only `SKILL.md` and the change record.

## Decisions

- **Change the description only** (Trigger: description; maintainer decision).
  - The description is the only text read before activation, so a skill that does not load can only be fixed there.
  - Rejected: adding discovery-path guidance to the body instead. The body is never read when the skill does not load, so it cannot close #82.
  - Rejected: also rewording `references/failure-modes.md` in this change ("trigger failures are always description failures"; nothing on discovery paths). That is a behaviour change of the loaded skill, which would need an added `Behavior:` requirement and outcome tests. It is proposed as a follow-up.
- **Candidate drafts, tried in the order B, A, C** (Trigger: description; maintainer decision).
  - The drafts are starting points for tuning, not approved wording. The shipped text is the first one that passes every case of the verification plan within the bounds of `## Description`, adjusted if needed without leaving them. Lengths are the folded values, trailing newline included.
  - **B (592)**, symptom first, and it names the move the solver made instead of loading: "Skill authoring — create, review, and repair Agent Skills. Use when a skill or agent instruction package never gets picked up, never triggers, fires on the wrong task, or gives inconsistent output — before moving or relinking it, since the usual cause is its description; when the request names an Agent Skill, SKILL.md, or an agent instruction package, or such a file appears in the material at hand; when drafting reusable instructions or a prompt package for an agent; or when deciding what belongs in a skill versus baseline agent docs. Not for application code or generic documentation."
  - **A (587)**, keeps the current order and rewrites the branch in place: "… when a skill or instruction package misbehaves — never gets picked up or triggers, fires on the wrong task, or produces inconsistent output — where the cause is usually its description, not its location; …".
  - **C (538)**, the smallest diff: ", including why one does not load" is added to the capability sentence, and "never gets picked up," to the symptom list. Nothing is reordered. It is the fallback.
  - Rejected: a synonym list ("not picked up, not detected, ignored, not activating"). `references/failure-modes.md` warns against synonym-stuffed triggers, and the list would exceed the budget.
  - Rejected: going past 600 characters. The skill's own linter would warn about its own description.
- **Restore the indirect-phrasing scenario removed in cb49b76 and add a direct-phrasing scenario, in the one modified block, and re-run every trigger case** (Trigger: description). The schema requires the whole block to be copied. `skill-authoring`'s testing reference asks for the complete affected evaluation to be re-run after a fix, so the three unchanged scenarios run again against the new text.
- **Add a non-skill near-miss** (Trigger: description, "Non-skill trigger (near-miss)"; maintainer decision). The rewrite makes "never triggers … fix it" more prominent, and a git hook shares that wording exactly. One run guards the `core` rule against over-broad descriptions.
  - Rejected: no new near-miss. The rewrite would widen the trigger surface untested.
  - Rejected: a `meta-harness` near-miss about harness drift ("our agents keep ignoring the conventions in AGENTS.md"). `great-skill-writing` legitimately claims "skill versus baseline agent docs", so that prompt has no single correct expectation.
- **The maintainer clears the stale global copy before the trigger tests** (Trigger: description; maintainer decision). The globally installed `great-skill-writer` carries the old description word for word, and the agent cannot write where it lives. Before the first round, the maintainer removes it or updates it to the current candidate. Removal is preferred. An updated copy has to be re-synced by the maintainer before every round, because each round tests an adjusted text, and the rubrics then count a load of either name (Verification plan).
  - Rejected: keeping the copy as it is and treating a load of only `great-skill-writer` as an invalid observation, with the degradation recorded. The copy would still sit in every solver's listing and compete in the very load decision under test. On a should-load case, loads of the old copy would use up the three attempts and leave the case skipped instead of decided. On a near-miss, its influence cannot be seen at all. Neither a pass nor a failure could then be attributed to the candidate text, and an installed user of this skill sees only one copy.
- **Acceptance on the hard fixture** (the two misbehaving-skill scenarios). The fixture's skill sits under `.agents/skills`, a directory Claude Code does not scan, as in #82.
  - Rejected: accepting on a fixture whose skill already sits under `.claude/skills`. That removes the plausible location cause and would certify an easier case than the one reported. That variant runs only as a diagnostic.
- **Sonnet-class solvers only** (all scenarios). This is the tier at which #82 failed and at which past changes certified this skill.
  - Rejected: escalating to a stronger tier. Acceptance would then cover only that tier and hide the reported failure.
- **Pass threshold 1/1 per case** (all scenarios; maintainer decision). This is the repository's convention (`lint-skill-help`) and keeps the fleet minimal.
  - Rejected: 2/2 on the two previously failing cases. It is more robust against sampling variance but costs two more runs per round.
- **README pair rows unchanged** (maintainer decision). They state the capability, not the triggers.
  - Rejected: appending "and diagnose skills that never load" to both rows. It adds nothing a reader choosing the skill needs, and no rule ties the rows to the description.

## Risks / Trade-offs

- [Over-triggering on non-skill "never triggers" requests (hooks, CI triggers, cron jobs, webhooks), which `skills/core/CONTEXT.md` warns pollutes every session] → the non-skill near-miss case; candidates keep "skill or agent instruction package" as the subject of the branch.
- [Pulling harness audits away from `meta-harness`, whose description claims project skills and agents missing conventions] → the rewrite adds nothing about harness drift; review watches the "instruction package" wording.
- [B's "the usual cause is its description" is only partly true for the fixture, where the location is also wrong] → the claim only steers the load decision. Diagnosing the location as well is the `failure-modes.md` follow-up.
- [A 1/1 result is one sample of a stochastic decision] → accepted as the repository's convention; the Validation section reports it as one attempt per case.
- [The stale `great-skill-writer` copy sits beside the candidate in every listing, so a load cannot be attributed to the new text] → the maintainer's precondition (Decisions), which P0 checks before each round.
- [The solvers' listing includes this repository's project-only skills. `skill-authoring` claims work under `.agents/skills/`, which T3 names and where F puts the skill, so a solver may load it instead of the target, a choice an installed user never faces] → recorded as an isolation degradation. Only a target load counts, and a failure of this kind is reported flagged (Verification plan, Isolation).
- [No candidate passes] → at most three rounds (B, A, C, each adjustable within the bounds). After that the agent stops and asks the maintainer to choose between narrowing the scenarios to what passes (recorded as a revision of this change) and accepting a documented limit.
- [Parallel trigger fleets of other changes share the session's skill listing] → this change's fleet runs alone.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Precondition (maintainer).** Before any trigger case runs, the stale globally installed `great-skill-writer` (user-level skills directory, the old description word for word) is removed, or updated to the current candidate text. If it is kept, the maintainer re-syncs it to the current candidate before every round. The agent cannot write there. Until this holds, the trigger cases wait; they are blocked, not skipped.

**Target.** "The target" is `great-skill-writing`, plus `great-skill-writer` when the maintainer kept an updated copy. A should-load case passes on a load of either name; a near-miss fails on a load of either name.

**Probe P0.** A clean-context subagent that sees no test prompt quotes the descriptions its session lists for `great-skill-writing` and `great-skill-writer`. It passes only when `great-skill-writing` shows the current candidate text word for word, and `great-skill-writer` is either not listed or shows exactly the same current candidate. P0 runs before each round, after any re-sync.

**Fixture F.** A throwaway git project under the session scratch directory, one fresh copy per case, outside this repository so solvers cannot answer from `.agents/knowledge/`:
- a small Python package (`pyproject.toml`, `src/<pkg>/__init__.py`);
- `RELEASE_CHECKLIST.md` and `CHANGELOG.md`;
- `.agents/skills/changelog-entries/SKILL.md` with `name: changelog-entries`, `description: Changelog helper.`, and a short body;
- `.pre-commit-config.yaml` with one local hook.

**Solver and observation.** One fresh clean-context Sonnet-class subagent per case, working directory its own copy of F. Prompts are in English, exactly as below; no prompt names the target. The observation is mechanical, the framework-native history that `skill-authoring`'s testing reference puts first: each solver's subagent transcript, which this harness keeps for every subagent, is grepped (never read whole) for a `Skill` tool call naming a target name or for `Launching skill:` with it. The loaded-skill list from the transcript is recorded for every case. The neutral `SKILLS_LOADED:` instruction from the same reference is appended to every prompt only as the fallback, read when a solver's transcript cannot be read. When both exist and disagree, the transcript wins and the disagreement is noted in Validation. One attempt; an attempt with neither a readable transcript nor a well-formed fallback line is invalid and retried, up to three attempts, after which the case is skipped and recorded as inadequate observability.

**Isolation.** Solvers see the skill listing of the session that dispatches them, not a candidate worktree. The candidate description must therefore be what that listing serves, which P0 confirms. If it is served from the repository checkout rather than a disposable candidate worktree, that is recorded as an isolation degradation. The same listing includes this repository's project-only skills, which an installed user never sees, and that is recorded as a second isolation degradation. `skill-authoring` matters most: its description claims work under `.agents/skills/` and says `great-skill-writing` fires together with it. Only a load of the target counts. Loading `skill-authoring` neither passes a should-load case nor fails a near-miss. A T2 or T3 failure in which the solver loaded `skill-authoring` still fails the round. Validation reports it as "failed, `skill-authoring` loaded instead of the target", with the solver's full load list, so the maintainer can tell a listing artifact from a description failure. D1 runs for it as for any T2 or T3 failure.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| Trigger: Skill authoring request | T1: "Write a SKILL.md so our agents follow the release checklist every time" | target loaded (C) | 1/1 | Sonnet-class | transcript grep for a target `Skill` call; `SKILLS_LOADED:` line only when no transcript can be read | fresh subagent, own copy of F, P0 passed |
| Trigger: Misbehaving skill, indirect phrasing | T2: "the instruction package I gave my agent for changelog entries never gets picked up — fix it" | target loaded (C) | 1/1 | same | same | same |
| Trigger: Misbehaving skill, direct phrasing | T3: "my changelog-entries skill never triggers when I ask for a changelog entry — its SKILL.md is under .agents/skills — fix it" | target loaded (C) | 1/1 | same | same | same |
| Trigger: Human documentation (near-miss) | T4: "write a README that explains how to run the release checklist" | target not loaded, under neither name (C) | 1/1 | same | same | same |
| Trigger: Human skills (near-miss) | T5: "which skills should a junior engineer build first?" | target not loaded, under neither name (C) | 1/1 | same | same | same |
| Trigger: Non-skill trigger (near-miss) | T6: "the pre-commit hook I added never triggers when I commit — fix it" | target not loaded, under neither name (C) | 1/1 | same | same | same |

A candidate passes when all six cases pass in one round. A failed case fails the round, and the next round re-runs all six against the adjusted description.

**Diagnostic D1 (not acceptance).** If T2 or T3 fails, its prompt runs once more in F′, which is F with the skill under `.claude/skills/changelog-entries/`. The result classifies the location confound for the maintainer and does not pass or fail the change.

Script and tool harnesses:
- `just check-skill skills/core/great-skill-writing`: no description error or warning.
- `uv run --offline skills/core/great-skill-writing/scripts/lint_skill.py --skill skills/core/great-skill-writing`: the OK line, with no warning above 600 characters.
- `just validate`, `just lint`, `just spec-validate`, `just check`.
- `git status --short`: only `SKILL.md` and `openspec/changes/misbehaving-skill-trigger/`.

Skipped:
- Baseline (RED) run: the #82 observations of 2026-09-10 (T2 and T3 not loading on the old text) are the baseline, which keeps the fleet minimal.
- Outcome tasks and grading: no `Behavior:` requirement changes; the body is untouched.
- `Script: lint_skill.py` scenarios: the script does not change.
