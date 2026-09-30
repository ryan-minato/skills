## Context

See proposal.md for motivation.

**Current shape** (line numbers on `main` at 10e29c0):
- **`scaffold-colab/SKILL.md`** (134 lines).
  - The exclusion is a loose bullet at :117-121, after the eleven workflow steps (:44-109) and the completion criterion (:111-115), just above `## Gotchas`.
  - No step names a commit, and nothing in the skill initializes a repository. Step 1 (:44-46) confirms the shape and "the current repository state", and step 2 (:47-57) writes the first files.
  - The loose bullet's only bound is "before the first commit". In a directory that is not yet a repository, the agent has to create one before it can commit, so the command runs after the repository exists.
  - The ephemeral-runtime shape reduces steps 2, 6, and 7 (:35-37) and leaves step 1 as it is.
  - Gotchas (:125-134): G1 local image is not Colab; G2 pip upgrades cascade; G3 regional registries are identical and the user's choice; G4 `# @param` over ipywidgets.
- **`scaffold-data-science/SKILL.md`** (161 lines).
  - The exclusion is the last bullet of `## Invariants` (:146-150), set apart by a blank line (:145).
  - Step 2 (:27-29) has two branches. An absent package is initialized with `uv init --package` and `uv.lock` is committed, the first commit the skill names. An existing project retains "its working package manager, lockfile, test tools, and quality gates": no `uv init` runs and no commit is named, so the build's first commit comes in a later step.
  - The skill's description covers "hardening an … early repository". Today the unconditional Invariants bullet covers both branches of step 2.
  - Gotchas (:154-161): G1 a branch or `latest` is not an identity; G2 a passing secret scanner does not prove the diff is free of PII, followed by a review procedure; G3 do not copy a template unchanged.
- **Reference shape.** `scaffold-ml` step 1 ends with the same procedure (:81-84). It landed in `ml-standard-alignment`.
- **Specs.** Neither skill has a domain. No existing scenario quotes the placement of the exclusion.

**Binding constraints:**
- **`skills/scaffold/CONTEXT.md`.** Disposable builders are never committed to the target repository. The marker string is fixed. No asset or generated file carries the marker.
- **Step numbers are referenced.** Colab's shape section names "steps 2, 6, and 7" and its step 11 names "step 8". Data-science step 12 names "step 10". A change that renumbers steps must rewrite each of these references.
- **Size.** Both bodies are far below 500 lines.
- **Scope.** The issue excludes `scaffold-ml`, the `meta` builders, and any Gotchas content beyond the review.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| colab — Behavior: Disposable builders stay out of every commit | `scaffold-colab/SKILL.md` `## Workflow` step 1: the procedure is appended to the step that confirms the repository state. When the target is not yet a repository, the same sentence directs running the procedure right after the repository exists and before the first commit. The loose bullet at :117-121 is deleted. | — |
| data-science — Behavior: Disposable builders stay out of every commit | `scaffold-data-science/SKILL.md` `## Workflow` step 2: the procedure is bound to the build's first commit on both branches of the step. In a new package it sits between `uv init --package` and committing `uv.lock`; in an existing project it runs before any commit. The `## Invariants` bullet at :146-150 and the blank line at :145 are deleted. | — |
| colab structure (no requirement) | `## Gotchas`: G1 and G4 stay; G2 and G3 are deleted. Step 8 (:94-96): the broken wrap is rejoined. | — |
| data-science structure (no requirement) | `## Gotchas`: G1 stays; G2 keeps only its first sentence; G3 is deleted. `## Invariants` (:138-140): the mis-indented continuation line is re-indented. Step 10 (:105-107): the broken wrap is rejoined. | — |

The requirements state only the ordering bound. That `SKILL.md` records the procedure inside a numbered workflow step is this placement, and the command proofs (the issue's acceptance) check it.

## External impact

None. The change touches no other skill, catalog README pair, symlink, `marketplace.json` entry, catalog `CONTEXT.md`, mirrored file, or harness file. Proof: `just validate`, and `git diff --name-only origin/main...HEAD` lists only the two `SKILL.md` files and this change directory.

## Decisions

- **Two new domains with ADDED Behavior requirements only** (both requirements; the maintainer's decision).
  - The move is where the exclusion runs, so the one behavior it touches is pinned. A later change cannot drop the procedure from the workflow unnoticed.
  - There is no `Trigger:` block because neither description changes. The precedent is `meta/meta-python-defaults`, which `management-code` created with `## Purpose` and Behavior requirements only. The schema's "exactly one per skill" (`schema.yaml:72`) is read as at most one; a later description change adds the block as ADDED.
  - Rejected: a `skip_specs` change. `schema.yaml:24-27` reserves `skip_specs` for a change to the repository itself.
  - Rejected: `Spec: none`. That is reserved for a change too small to plan, such as a pin bump or a typo, and the Gotchas review is more than that.
  - Rejected: backfilling a `Trigger:` block for each domain. That specifies a description nobody is changing.
- **The SHALL statements carry the ordering bound, not the workflow-step placement** (both requirements). The maintainer's recording, "a workflow step that happens before the first commit", is honored in Placement and in the issue-acceptance proof. `schema.yaml:53-54` says a spec "never describes the diff, the file layout, or the wording of SKILL.md", and no scenario can observe which section of `SKILL.md` holds a sentence.
  - Rejected: "in a workflow step that runs before the build's first commit" in the SHALL. It describes the layout of `SKILL.md`, and only a command proof could check it.
- **colab: workflow step 1** (colab requirement; the maintainer's decision).
  - Step 1 inspects the repository state, and nothing is written before step 2. The ephemeral-runtime shape leaves step 1 as it is, so both shapes reach the procedure. It also matches `scaffold-ml`.
  - Step 1 runs before anything is written, so in a directory that is not yet a repository `git rev-parse --git-path info/exclude` exits 128 there (checked on 2026-09-30). A bare move would create an ordering hazard that the loose bullet, bound only to the first commit, does not have. The step-1 sentence therefore carries the condition: when the target is not yet a repository, run the procedure right after the repository exists and before the first commit ("Directory not yet a repository").
  - Rejected: step 1 worded as `scaffold-ml` words it, with no condition. It inherits the hazard the `scaffold-ml` follow-up is meant to remove.
  - Rejected: step 2, the layout step. A sentence about commits reads as layout there, and the ephemeral shape reduces that step.
  - Rejected: a new step of its own. It would renumber steps and the three step references listed under Context.
- **data-science: workflow step 2, bound to the build's first commit on both of its branches** (data-science requirement, all three scenarios; the maintainer's decision, right before `uv.lock` is committed, extended to the existing-project branch).
  - `uv init --package` may be what creates the repository. Checked on 2026-09-30 with uv 0.12.5: `uv init --package` in a non-git directory creates `.git`, and `git rev-parse --git-path info/exclude` exits 128 outside a repository. Step 2 is the first place where the command can run, and the last place before the `uv.lock` commit.
  - The existing-project branch of step 2 runs no `uv init` and commits no `uv.lock`, so the sentence states the bound for it as "before any commit" ("First commit of the build").
  - Rejected: binding the procedure to the `uv.lock` commit alone. Once the Invariants bullet is deleted, an existing project would reach its first commit with no exclusion.
  - Rejected: step 1, the inventory, for literal parity with `scaffold-ml`. It puts the command before the repository can exist.
- **The procedure's content moves unchanged; its sentence is fitted to its step** (both requirements).
  - Four elements stay verbatim: the marker string, `$(git rev-parse --git-path info/exclude)`, staging explicit paths, and reading `git status` before each commit. So does the bound "before the first commit": colab conditions it on a repository existing, and data-science states it for each branch of step 2.
  - The lead sentence is reworded into a step's voice.
  - Rejected: moving the loose bullet byte for byte. Its lead "Disposable builders never enter a commit:" is a rule's voice, not a step's, and neither step's ordering (a repository that may not exist yet; the `uv.lock` commit) fits it.
- **The Gotchas bar: a non-obvious fact that no step or reference states before the wrong assumption is made** (the structure rows; the maintainer's decision). This is the bar of `.agents/knowledge/skill-quality.md` (Instruction patterns), read as `ml-catalog-asset-cleanup` read it.
  - colab G1 stays. `references/local-runtime.md:5-7` says only that a real Colab run confirms. The list of Colab-only features (Drive mounting, Google auth, form rendering) is stated nowhere else.
  - colab G2 goes. `references/environment-debugging.md:10-14` states the cascade, and step 3 loads that reference "before adding, pinning, or upgrading any package".
  - colab G3 goes. Step 4 states it (identical images, "let the user decide"), and so does `references/local-runtime.md:16-18`.
  - colab G4 stays. `references/colab-forms.md:5-8` and `assets/agents-md.md:45-46` state it, but step 6 loads `colab-forms.md` only "when adding parameters, cell titles, or hidden-code forms", after the choice between forms and ipywidgets is made.
  - data-science G1 stays. `references/storage-huggingface.md:14-15`, `storage-s3.md:14-16`, and `model-inference.md:6-7` state it per backend. A source outside those references, such as an HTTP download or git-hosted data acquired under `src/<package>/sources/`, has only this bullet.
  - data-science G2 is trimmed to its first sentence, the fact (the maintainer's decision). The review procedure it adds is also in `assets/base/agents-md.md:60-65`, which step 3 deposits as `AGENTS.md`. That copy addresses the target project's future agents, and it is written after step 2, so it does not carry the procedure for the builder's own commits before step 3. That loss is a risk below, not coverage.
  - data-science G3 goes. Step 3 says "Copy and rework every base asset; replace every `__UPPER_CASE__` placeholder", the completion criterion requires every placeholder resolved, and `scripts/validate_scaffold.py` (`PLACEHOLDER_RE`, run by step 11) fails on one.
  - Rejected: deleting every bullet stated anywhere else. That drops colab G4 and data-science G1, the two guards before the wrong assumption.
  - Rejected: leaving the Gotchas untouched. The issue's outcome requires the review.
- **The adjacent whitespace defects are fixed in the same commit** (the maintainer's decision). They are source-only, and Markdown renders the soft breaks and the continuation line the same either way. Rejected: leaving them for a later change, which would touch the same sections again.
- **Commit type `refactor`, one commit for both skills**, scoped `scaffold-colab, scaffold-data-science`. No installed behavior was wrong, and none is added. Rejected: `fix`, since the procedure already existed with the same bound. Rejected: `docs`, since the installed `SKILL.md` files change.
- **`scaffold-ml` is a follow-up** (the maintainer's decision). Its step 1 runs the command during the inventory, before a repository may exist. The issue excludes it.

**Decisions for the maintainer to confirm on the draft** (the agent's own readings, not settled by the maintainer):
- Reading `schema.yaml:72` ("exactly one per skill") as at most one, so a domain may exist without a `Trigger:` block; and whether that line should be clarified in a separate repository change.
- Keeping colab G1 (local image is not Colab), which the maintainer's Gotchas list does not mention.
- Rewording the procedure's lead sentence into a step's voice, with the four elements verbatim.
- The colab step-1 condition for a directory that is not yet a repository, instead of a bare move.
- Extending the data-science bound to the existing-project branch of step 2 ("before any commit").
- Keeping "workflow step" out of the SHALL statements and in Placement and the issue-acceptance proof.

## Risks / Trade-offs

- **[The move is read as a behavior change]** → The bound "before the first commit" stays. Neither new position is later than the first commit the skill can make on any path, and neither runs the command before a repository exists. The readback checks both requirements against the finished text.
- **[A deleted Gotcha was the only statement an agent sees on some branch]** → For each deleted or trimmed bullet, the readback quotes the step or reference that now carries the fact, and the sentence that loads it, and confirms the load fires before the fact is needed.
- **[colab in a directory that is not yet a repository]** → Moving the procedure into step 1 creates this ordering hazard: step 1 runs before any repository exists, and the command exits 128 there. The step-1 sentence carries the condition (run it right after the repository exists and before the first commit), the colab "Directory not yet a repository" scenario pins it, and O4 exercises it in a fixture that is not a repository.
- **[An existing data-science project reaches its first commit without the exclusion]** → The step-2 sentence states the bound for the existing-project branch too, the "First commit of the build" scenario pins it, and O3 exercises it in an existing repository.
- **[The builder's own commits before step 3 lose the review sentence]** → Trimming data-science G2 removes the only statement of the review procedure that the builder reads before step 3 deposits `AGENTS.md`. In a new package, the one commit before step 3 carries `uv.lock` and the files `uv init --package` generates (`pyproject.toml`, `.gitignore`, `.python-version`, `README.md`, a package `__init__.py`), no user data. In an existing project, the skill names no commit before step 3. The kept fact still warns that a passing scanner is not proof. R1 grades explicitly whether the kept fact suffices before the builder's first commit on each branch of step 2, and a GAP there reopens the trim before implementation is accepted.
- **[A domain without a `Trigger:` block reads as incomplete]** → Each `## Purpose` names the narrow scope, and the precedent is recorded above.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Solver tier:** Sonnet-class, the least capable tier the scaffold builders were certified at. No description changes, so there are no trigger cases, and each solver prompt names its skill.

**Observation:**
- The solver's transcript: the staging commands, the `git status` reads, the exclude write, and their order relative to each `git commit`.
- The fixture's final state: `cat "$(git rev-parse --git-path info/exclude)"`, and `git log --name-only --format=` over the build's commits.

**Isolation:**
- One fresh clean-context solver per case, and one throwaway fixture per case under the session scratch directory.
- The candidate skill is copied into the user skills directory for the run and removed afterwards.
- Local work only: no push and no remote write. The solver stops after the step named in the prompt.
- One attempt, and up to three when the observation is invalid.
- An independent clean-context grader on a capable model grades the anonymized outputs.

**Fixtures:**
- `colab-repo`: a git repository with no commits. Its `.claude/skills/` holds copies of `scaffold-colab` and two `meta` builders, all carrying the marker, and one durable stub skill whose description does not carry it.
- `colab-empty`: a directory that is not a git repository, with the same four skill directories as `colab-repo`.
- `ds-empty`: a directory that is not a git repository and holds no package, with the same four skill directories (`scaffold-data-science` in place of `scaffold-colab`).
- `ds-repo`: a git repository with one commit holding an existing uv package `sales_insights` (`pyproject.toml`, `uv.lock`, `.gitignore`, `src/sales_insights/__init__.py`). One local CSV sits under `data/raw/`, ignored by that `.gitignore`. The same four skill directories as `ds-empty` are present and untracked.

Each rubric item scores 1. Critical items are marked (C), and any critical failure fails the case.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| colab: First commit of a notebook project; Durable skill beside the builders | O1 in `colab-repo`: "Use the scaffold-colab skill to set this repository up as a Colab notebook project: one CPU tutorial notebook on pandas basics, cell outputs not committed. Decided already: registry region `us`, no google-colab-cli, and I decline the meta handoffs. Local file changes are approved. Commit the initial layout, then stop — do not build images, connect to Colab, install anything, or push." | the exclude file lists the three builder directories before the first `git commit` (C); no commit of the build contains a path under a builder directory (C); the durable stub's directory is absent from the exclude file (C); every file is staged by explicit path, with no `git add -A`, `.`, or `--all`; `git status` is read before each commit | all (C) and ≥ 4/5 | Sonnet-class | transcript + fixture state | as above |
| data-science: Directory not yet a repository; Durable skill beside the builders | O2 in `ds-empty`: "Use the scaffold-data-science skill to start this directory as a data-science project: package `sales_insights`, one local CSV source, one local Parquet product, no model, a Markdown report. I decline the meta handoffs. Local file changes are approved. Carry the build through the commit of `uv.lock`, then stop — do not push." | the exclude entries are written once the directory is a repository and before the commit carrying `uv.lock`, and no exclude command fails for want of a repository (C); no commit of the build contains a path under a builder directory (C); the durable stub's directory is absent from the exclude file (C); every file is staged by explicit path; `git status` is read before each commit | all (C) and ≥ 4/5 | same | same | same |
| data-science: First commit of the build; Durable skill beside the builders | O3 in `ds-repo`: "Use the scaffold-data-science skill to harden this existing data-science project: keep its uv package `sales_insights`, add one local Parquet product built from the CSV, no model, a Markdown report. I decline the meta handoffs. Local file changes are approved. Lay out the package structure, commit it, then stop — do not push." | the exclude file lists the three builder directories before the build's first `git commit` (C); no commit of the build contains a path under a builder directory (C); the durable stub's directory is absent from the exclude file (C); every file is staged by explicit path, with no `git add -A`, `.`, or `--all`; `git status` is read before each commit | all (C) and ≥ 4/5 | same | same | same |
| colab: Directory not yet a repository; Durable skill beside the builders | O4 in `colab-empty`: "Use the scaffold-colab skill to start this directory as a Colab notebook project: one CPU tutorial notebook on pandas basics, cell outputs not committed. Decided already: registry region `us`, no google-colab-cli, and I decline the meta handoffs. Local file changes are approved. Commit the initial layout, then stop — do not build images, connect to Colab, install anything, or push." | the exclude entries are written once the directory is a repository and before the build's first `git commit`, and no exclude command fails for want of a repository (C); no commit of the build contains a path under a builder directory (C); the durable stub's directory is absent from the exclude file (C); every file is staged by explicit path; `git status` is read before each commit | all (C) and ≥ 4/5 | same | same | same |

**Readback case.** R1 is one clean-context subagent that reads both finished `SKILL.md` files and their references.
- For every scenario, it quotes the passage that produces the scenario's THEN and grades it PRESENT, WEAK, or GAP.
- For colab "Ephemeral runtime", it also confirms that the shape section does not reduce the step that carries the procedure.
- For each deleted or trimmed Gotchas bullet (colab G2 and G3, data-science G2 and G3), it quotes the step or reference that now carries the fact and the sentence that loads that reference, and says whether the load fires before the fact is needed.
- For data-science G2 in particular, it grades whether the kept fact alone guards the builder's first commit on each branch of step 2, before step 3 deposits `AGENTS.md`.
- Any GAP is a critical failure. Threshold: no GAP.

**Command proofs:**
- The issue's acceptance: `grep -n "info/exclude" skills/scaffold/scaffold-colab/SKILL.md skills/scaffold/scaffold-data-science/SKILL.md` prints one hit per file. An `awk` bounds check places each hit between `## Workflow` and the completion criterion, inside colab step 1 and data-science step 2. This proves the workflow-step placement that the requirements leave to Placement.
- Each of the four verbatim elements (the marker string, the exclude command, explicit paths, `git status`) appears inside that step.
- `git diff --name-only origin/main...HEAD` lists only the two `SKILL.md` files and this change directory.
- `just check-skill skills/scaffold/scaffold-colab skills/scaffold/scaffold-data-science`, `just spec-validate`, and `just check` pass.

**Skipped:**
- Trigger cases, because no description changes.
- An outcome run for colab "Ephemeral runtime". The procedure sits in a step that both shapes share, and R1 checks that. A fifth solver would exercise the same text.

## Open Questions

None.
