## Context

See proposal.md for motivation. The companion repository change `management-code-harness` carries this repository's own scripts, checks, knowledge, and catalog rules.

**Current shape:**
- **The sdd framework skills.** `openspec-workflow` and `spec-kit-workflow` ship one bundled script each: `scripts/spec_changes.py` (seven subcommands) and `scripts/spec_kit_features.py` (six). The agent runs it locally, for example the archive executor, and the delivered workflows call a byte-identical copy of it at the project's `scripts/`.
  - Both `references/github.md` and `references/gitlab.md` say "byte-identical copy", and every asset header repeats it.
  - The comment-command workflow wraps the script in `{ … } > reply.md 2>&1 || true`, so a crash posts a traceback and the run stays green.
  - The GitLab `show`, `status`, and `labels` jobs pipe the script through `tee`. GitLab Runner's generated job script turns on `pipefail` where the shell supports it and always `errexit` (read from the runner's bash shell writer on 2026-09-29), so a failing script already fails the job and its output stays in the log. The OpenSpec fragment runs `python3` and `jq` on `{{NODE_IMAGE}}`, which a slim Node image does not carry.
- **`meta-github-workflow`.** It delivers:
  - `assets/check_commits.py` and `assets/check_taxonomy.py`. The taxonomy check installs PyYAML with `pip` in CI, and its `as_dict`/`as_list` helpers turn a malformed `release.yml` into "no references".
  - its own `scripts/sync_labels.py`, beside `labels.json`;
  - `scripts/run_log_digest.py`, `scripts/next_version.py`, and `scripts/project_fields.py`, into the durable project skill. The builder runs `sync_labels.py` and `run_log_digest.py` itself; `next_version.py` and `project_fields.py` exist only to be copied.
  - its workflow assets, which set no shell, so `run:` steps use the Actions default `bash -e {0}`. Under that default the aggregator gate's `printf … | grep -Eq` is correct: the pipe's status is `grep`'s.
- **`meta-gitlab-workflow`.** It delivers `assets/check_commits.py`, byte-identical to GitHub's. Into the project skill it delivers `scripts/pipeline_log_digest.py` and `scripts/next_version.py`; the builder never runs `next_version.py` itself.
- **`meta-harness-architecture`.** Its only guidance on scripts is "Custom checks must explain what failed, why it matters, and the likely fix" (`## Using the assets`, and `references/layers.md`).
- **`meta-python-defaults`.** It mentions PEP 723 and `uv run` once, for a one-off script, in `references/dependency-managers.md`.
- **`scaffold-ml`.** The `docker-build` recipe in `references/containers.md` tests `[ -n "$(git status --porcelain)" ]` inside an `if` and passes `$(git rev-parse HEAD)` as a build argument. A failed git read is therefore treated as a clean tree and an empty commit.

**Binding constraints:**
- **Methodology mirror.** `## Harness Methodology` is byte-identical in `core/meta-harness` and `meta-harness-architecture`, and `scripts/validate_skills.py` enforces it: edit the architecture source first, then copy it.
- **Catalog rules.**
  - `meta/CONTEXT.md` says "Assets are raw starting shapes. Rework every line". The companion change adds the carve-out for script assets, which are working code whose marked settings are configured.
  - No asset may carry the disposable-builder marker.
  - `sdd/CONTEXT.md`: automation is fork-safe by construction, and a script pins the tool flags it relies on with the verification date.
- **Self-containment.** No path leaves a skill directory. Handoffs route through `ryan-minato-skills-installing`. Descriptions do not change; each is within budget.
- **Size.** SKILL.md bodies stay under 500 lines. Detail goes to references with a load sentence.
- **Out of scope.** Public skills' own `scripts/` keep their rules and contents.
- **Assets.** Every script asset is Python, standard library first, and runs on 3.10 or later, the floor the user set. The taxonomy check alone carries a third-party dependency (PyYAML), declared in PEP 723.

**Precedent.** No asset script has had a `Script:` requirement so far: `check_commits.py`, `check_taxonomy.py`, and `run_manifest.py` have none. The sdd CI assets get one, because the fork safety of the delivered workflows rests on them. The meta assets are specified through Behavior requirements and verified by the script harness below.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| MH — Behavior: Management code belongs to the project | `core/meta-harness/SKILL.md` `## Harness Methodology` › `### Write for agents and progressive loading`: one sentence after "put deterministic repeated logic in a script". It is copied byte-identically from the architecture source. | — |
| MHA — Behavior: …written for its role and owned by the target | the same methodology sentence in `meta-harness-architecture/SKILL.md`. `## Using the assets`: the paragraph "For scripts, tests, linters, CI…" becomes a short statement of the rules, pointing at the new reference. | "Before writing or depositing any script, hook, CI step, task-runner recipe, or project-skill script, read [references/management-code.md](references/management-code.md)." |
| MHA — Behavior: …language the project's community scripts in | `references/management-code.md` `## Language` (the judgment, Go/`go run` and C/C++ as the two anchors, "check current official documentation for a borderline ecosystem") | as above |
| MHA — Behavior: …dependencies and invocation… | `references/management-code.md` `## Dependencies and invocation`: self-contained versus ecosystem convention; the test that a standard-library version would need its own tests, so uv is recommended; when `uv run` applies; recording the language and invocation in the target's knowledge. A Python project's idiom is `meta-python-defaults`' job. | as above |
| MHA — Behavior: …follows the fail-fast philosophy… | `references/management-code.md` `## Failing fast`: fail-fast as a design philosophy with one firm outcome, no hidden failure; the interface checks to keep or add; the fallbacks to remove; failing at the call or after collecting findings, as a judgment; documented nulls as domain logic; what is not defense (structural safety, idempotence, narrow retries). `references/layers.md` "Custom checks must fail with messages…" gains: a finding points at the file to fix, never a traceback. | as above |
| MHA — Behavior: Shell in management code… | `references/management-code.md` `## Shell`: the ways a failure gets lost (a command tested inside a condition, `\|\| true`, a piped status, a sourced script's flags) and the ways to keep it (not piping, checking a status before the step ends, strict mode); the Actions default shells for reference; strict mode and `defaults.run.shell: bash` as tools, not rules. `references/project-skill.md`: scripts in a project skill are management code. | as above |
| MPD — Behavior: …declare dependencies inline… | new `references/management-scripts.md` `## Dependencies and invocation`; `references/dependency-managers.md` "One-off script" links to it | step 5 of SKILL.md: "Read [references/management-scripts.md](references/management-scripts.md) when the project has, or the harness build adds, scripts that never ship in the product — checks, hooks, CI or administration scripts." |
| MPD — Behavior: …check their interfaces… | `references/management-scripts.md` `## Error handling`: the idiom, with one short before-and-after example | as above |
| MPD — Behavior: Locking a CI script's dependencies… | `references/management-scripts.md` `## Locking dependencies in CI`: the two options, the risk table (privileged trigger, writable token, or secrets → recommend; read-only job without secrets → optional), and "present, then wait for the decision" | as above |
| MGH — Behavior: Scripts delivered… from management assets | SKILL.md step 4: the script sentences at lines 219–233, and a rule that script assets are working code whose marked settings alone change. New `assets/sync_labels.py` and `assets/run_log_digest.py`. `assets/next_version.py` and `assets/project_fields.py`, moved from `scripts/`. `references/planning-and-goals.md`, `projects-v2.md`, `releases-deploy-registry.md`, `actions-and-checks.md` (the builder keeps using its own digest), `durable-harness.md` (default structure lists `scripts/`; the synchronization section says delivered scripts are owned, not mirrored), `assets/project-skill.md` (the digest slot names the project path). SKILL.md step 5: no delivered rule, check, or instruction binds a delivered script to a builder script. | existing load sentences |
| MGH — Behavior: Delivered workflows and the taxonomy check… | every `run:` step of `assets/workflow-*.yml` is read for a failure it could lose, and only such a step is rewritten, with no shell default imposed; the inline scripts of `workflow-pr-checklist.yml` and `workflow-triage.yml` drop fallbacks on data the API guarantees and fail on an unexpected shape; `assets/check_taxonomy.py` gets a PEP 723 header, `assets/workflow-taxonomy.yml` uses setup-uv and `uv run`; `references/commits-and-contributions.md` and the `compatibility` field describe the uv prerequisite and point locking at the risk rule | existing load sentences |
| MGL — Behavior: Scripts delivered… from management assets | SKILL.md step 4 (lines 161–168); new `assets/pipeline_log_digest.py`; `assets/next_version.py` moved from `scripts/`; `references/commits-and-contributions.md`, `durable-harness.md`, `assets/project-skill.md`; step 5's disposal test | existing load sentences |
| MGL — Behavior: Delivered job scripts do not lose a failure | `references/ci-and-runners.md`: a job script ends failed when a command it depends on failed, at the command or when the script ends. GitLab Runner already sets `errexit`, and `pipefail` where the shell supports it, so the rule mostly concerns commands tested inside a condition, `\|\| true`, and a shell without `pipefail` (verified against the runner source with its date); `references/commits-and-contributions.md`: the commit-check job shape | existing load sentences |
| OSW — MODIFIED installation requirement | `references/github.md` and `gitlab.md` `## What lands where` (the script row names the asset and the project path, with no copy); `## Fork safety`; `## Verification after installing`. SKILL.md `## Installing the automation`. SKILL.md `## Ready, then the freeze`: the archive command names the skill's own script by its skill directory, and says the project's `scripts/spec_changes.py` is the CI script and has no `archive`. Every asset: header, `run:` lines, and the reply status handling in `workflow-spec-command.yml`. `assets/gitlab/ci-spec-jobs.yml`: a before-script precondition that names a missing `python3` or `jq` in the image; the `tee` pipes stay, since the runner's shell options already fail the job. | existing load sentences |
| OSW — Script: assets/spec_changes.py | new `assets/spec_changes.py` | — |
| SKW — MODIFIED installation requirement; Script: assets/spec_kit_features.py | the same places in `spec-kit-workflow`; new `assets/spec_kit_features.py` | existing load sentences |
| SML — MODIFIED container recipe requirement | `references/containers.md` `## Task-runner recipes`: `docker-build` reads the tree status and the commit before testing them and stops when either read fails; strict mode is one way to write it | existing load sentence |

## Dependencies and handoffs

No new handoff and no new skill name in any installed skill. `meta-harness-architecture` may say that a Python project's idiom comes from `meta-python-defaults`: they are sibling `meta` builders, and the catalog grants dependencies between them. `meta-harness` names no skill.

## External impact

- **`skills/meta/meta-spec-workflow/assets/spec-workflow.md` and `SKILL.md`.** "owns the request automation and its script" becomes "supplies the request automation and its management script; the project owns the deposited copy". Proof: readback. The wording changes and the behavior does not, so there is no delta. The precedent is `ml-catalog-asset-cleanup`.
- **Companion change `management-code-harness`:**
  - the `meta/CONTEXT.md` carve-out for script assets;
  - the lint scope that covers `skills/meta/*/assets` and `skills/sdd/*/assets` (the global py310 target already matches the assets' 3.10 floor);
  - the register row that pairs the sdd GitHub workflow assets with this repository's workflows. The workflows keep the same steps and the same script path, and the row gains the pair it relied on implicitly: the asset's CI-facing CLI and this repository's `scripts/spec_changes.py`.

  Proof: `just check`, and the companion's verification plan.
- **No change to** symlinks, `marketplace.json`, descriptions, or the `sdd` README rows, which say the skills ship `spec_changes.py` and `spec_kit_features.py`. Proof: `just validate`.

## Decisions

- **A management asset written for the role, never a binding** (all "delivered from management assets" requirements; OSW and SKW installation).
  - A management asset is written for the job that calls it, the project owns it, and fixes to the skill do not reach it.
  - The two scripts need not be identical, and nothing keeps them identical. Matching content is allowed where the role happens to need the same code, so no check or rubric treats identity itself as a defect; the defect is a rule, check, or instruction that binds them.
  - Rejected: keeping the copies with a divergence marker. That leaves the binding in place and the defensive style with it.
- **Scripts the builder never runs move to `assets/`; scripts it also runs get a second, separate asset** (MGH, MGL).
  - `next_version.py` and `project_fields.py` exist only to be delivered: moved, and rewritten by R5.
  - `sync_labels.py`, `run_log_digest.py`, and `pipeline_log_digest.py` stay in `scripts/` for the builder and get separate assets for the target.
  - Rejected: a second asset beside a script that nothing else uses (two files for one job).
  - Rejected: dropping the digest in favor of a documented one-line command. `gh run view --log-failed | tail` keeps only the last failed job, and the per-job tail is the point of the digest.
- **The sdd CI assets hold only what the workflows call, and keep the path the workflows call** (OSW, SKW; the subset is the user's decision).
  - `assets/spec_changes.py` and `assets/spec_kit_features.py` offer `snapshot`, `check`, `show`, `status`, and `labels`, the last with `--taxonomy`. They keep both head sources, because the GitLab jobs use the git source.
  - They have no `archive` and no `related`: the archive executor runs the skill's bundled script or the tool, never CI.
  - They are deposited at the project's `scripts/spec_changes.py` and `scripts/spec_kit_features.py`. The workflow assets, this repository's workflows, and projects that installed the automation earlier all call those paths, and R1 asks only that nothing binds the deposit to the bundled script, not that its name differ.
  - The one hazard of the shared name, `python3 scripts/spec_changes.py archive` run from a target's root reaching the project's copy, ends at once with exit 2 and argparse's list of valid subcommands. SKILL.md names the skill's own script for the archive and says the project's copy has none.
  - Each asset's requirement is named by its path (`Script: assets/spec_changes.py`), because the bundled script's `Script: spec_changes.py` shares the file name in the same domain.
  - Rejected: a new file name such as `spec_request.py`. Every workflow asset, both references, this repository's workflows, and earlier adopters would change path to prevent a mistake that already fails fast.
  - Rejected: a full-featured copy (the size, and a second archive implementation to keep correct).
- **The CI-facing CLI matches this repository's own script** (OSW, SKW).
  - Same subcommand names, flags, and output, so the asset workflows and this repository's workflows keep the same steps and the same script path.
  - The companion registers that pair. It is a documentation mirror between two management scripts, not a binding to a product script.
- **Assets are Python 3.10+, standard library first, delivered as they are** (all asset requirements; user decision).
  - R2's project-language rule governs code the agent writes. A tested asset is delivered in Python, and the target may port its copy.
  - Rejected: porting at delivery time, because an untested port can drop fork-safety properties.
  - Rejected: one asset per ecosystem, which doubles the maintenance and the tests.
- **Fail-fast is a design philosophy, not a fixed procedure** (MH, MHA fail-fast and shell, MPD interfaces, MGH and MGL workflows; user clarification).
  - Its one firm outcome is that a failure that matters is never hidden: never turned into a pass, a default, or a wrong answer, and nothing is built on a result known to be bad.
  - When the failure surfaces is a judgment. Stopping at the failing call, collecting findings and failing at the end of a script, and checking a status before an Actions step ends all fit.
  - Rejected: strict mode and a bash default shell as rules for every step. They are tools; a step that checks its status before it ends is just as correct, and forcing strict mode rewrites working steps for no gain.
  - Rejected: stopping every check at its first finding. A check that reports every violating file in one run serves the person fixing them better.
- **Readability first; interfaces are checked; failures are not hidden** (MHA fail-fast, MPD interfaces, the sdd Script requirements; user clarification).
  - A check at an external interface that fails with a named message is fail-fast and stays. What goes is a default standing in for a failure.
  - Rejected: removing every `isinstance` or parse check "because the API guarantees it". That turns a changed interface into a traceback far from its cause, or into silent wrong output.
- **A comment command's bad argument is exit 2 and a green run; every other failure posts its output and fails the run** (OSW, SKW; user decision).
  - A commenter's typo is answered in the reply, and a crash is visible to the maintainer.
  - Rejected: always failing the run, which makes a typo look like an outage.
  - Rejected: `|| true`, which hides crashes.
- **The methodology gets one sentence, and the rules get a reference in the architecture builder** (MH, MHA; user decision).
  - The durable `meta-harness` carries the principle after the builders are disposed. The builder carries the detail it applies at build time.
  - Rejected: no methodology change. The rule would then disappear with the disposable builder.
- **Locking is the user's decision, defaulted by the job's actual risk** (MPD locking, MGH taxonomy check; user decision).
  - Rejected: always lock, which adds a lockfile for a read-only check.
  - Rejected: never mention it, which leaves a privileged job resolving dependencies at run time.
- **No lint rules in `meta-python-defaults`** (user decision). R5 is stated as the idiom there, with blind, silent, and restating handlers allowed only with a stated reason; this repository enforces it with ruff in the companion change.
- **Classification of the scaffold assets** (non-goals).
  - Product code, untouched: `local-data-guard.py` (the pipeline calls it around production runs, and its manifest ships with the product), `run_manifest.py`, `stages.py`, and `torch_health.py`.
  - Management code that already conforms: the scaffold justfiles, apart from `scaffold-ml`'s `docker-build`, which this change fixes.

## Risks / Trade-offs

- **[A delivered asset drifts from the skill's script, and fixes no longer propagate]** → Accepted by R1. The asset is tested here, the target owns its copy afterwards, and the handoff says so.
- **[The sdd CI asset carries fork-safety properties that the target may edit away]** → The `Script:` scenarios pin them in the asset. The workflow structure stays the first line of defense: base checkout only, least permissions, collaborator-only commands, and the literal label taxonomy.
- **[Fail-fast over-applied: strict mode imposed everywhere, checks that stop at their first finding, a guard before every line]** → The reference states the philosophy and its one firm outcome, and shows a status checked before a step ends and a check that collects its findings. The outcome rubrics treat both over-application and hidden failures as failures.
- **[Reading R5 as "drop checks" removes a needed one]** → The asset rewrites list each external interface and what is checked there. The harness feeds each interface a malformed response and expects a named failure with nothing built on the bad response.
- **[Floor-tier solvers read "fail fast" as "no validation" or as "wrap everything in try"]** → The references carry one short before-and-after example. The outcome rubrics mark both failure modes critical.
- **[The GitLab fragments cannot run live here]** → YAML parse, each job's failure path read, and a local run of each job's script with stubbed tools.
- **[uv facts, such as `uv lock --script` and `exclude-newer` in a script's `[tool.uv]` table, move between releases]** → They are verified against uv's current documentation when the reference is written, and the verification date is recorded next to them.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Solver tier:** Sonnet-class, the least capable tier past changes certified these skills at.

**Observation:**
- Loading: the framework's native skill-load history, otherwise the neutral `SKILLS_LOADED:` self-report.
- Outcomes: the transcript plus the fixture's diff against its initial commit.
- Delivered files: `cmp` against the asset they come from, after its marked settings.

**Isolation:**
- One fresh clean-context subagent per case.
- Each fixture is a throwaway git project under the session scratch directory.
- The candidate skills are copied into the user skills directory for the run and removed afterwards.
- One attempt, and up to three when the observation is invalid.
- An independent clean-context grader on a capable model grades anonymized outputs.

No description changes, so there are no trigger cases.

Each rubric item scores 1. Critical items are marked (C), and any critical failure fails the case.

**Fixtures:**
- `go-orders`: a Go module with `migrations/NNN_*.up.sql` files, one of which lacks its `.down.sql`, and a GitHub remote.
- `cpp-sensor`: a CMake project with `config/*.yaml`, `docs/config.md` stating two rules, and no uv anywhere.
- `web-shop`: a TypeScript project with `package.json` and a GitHub remote. It has an installed project skill whose bundled `scripts/pr_body_check.py` checks a pull request body, and `.github/pr-rules.yaml`.
- `py-labels`: a Python project whose dev container and CI both install uv, with a `pull_request_target` labeling workflow and a `pull_request` check.
- `gh-base`: a GitHub repository with a `checks` workflow and gate, a `labels.json`, and knowledge recording an approved design: taxonomy approved, SemVer not chosen, Projects not opted into, Actions diagnosis selected, developers without uv. Variants: `gh-openspec`, adding an `openspec/` directory; `gh-speckit`, adding `specs/001-export/`.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| MH: Skill script offered as the project's check | O1, in `web-shop` with only `meta-harness` among the candidates: "Add a CI check that fails when a pull request body has no Testing section. The pr-body-check skill already has a script for it." | a project-owned script under the project's own scripts (C); no rule, check, or instruction keeping it identical to the skill's script, with matching content alone not counted as a failure (C); the failure message names the missing section | all (C) and ≥ 2/3 | Sonnet-class | transcript + diff + `cmp` | as above |
| MHA: Go project; Command inside a condition; Validator finding; Invocation recorded | O2 in `go-orders`: "Set up this service's agent harness: a CI workflow with a check that every up migration has a down migration, and record how our scripts run." | check written in Go and run with `go run` (C); a failure of any command the workflow depends on fails its step, and none is read as "nothing to check" (C); the finding names the migration file missing its pair and prints no traceback; the harness knowledge names the language and the command | all (C) and ≥ 3/4 | same | transcript + diff | same |
| MHA: C++ project; YAML parsing without uv; Validator finding; Findings collected before failing | O3 in `cpp-sensor`: "Add a CI check and a pre-commit hook that validate config/*.yaml against the rules in docs/config.md, and report every bad file in one run." | script in Python or Deno, not C++ or a long Bash script (C); uv with a PEP 723 header recommended instead of a hand-written YAML parser (C); no `uv run` written for an environment without uv (C); every violating file reported in one run, then one non-zero exit, not a stop at the first finding (C); each violation names the file and the rule | all (C) and ≥ 4/5 | same | transcript + diff | same |
| MHA: Bundled script available; Node project dependency; External command output; Documented null | O4 in `web-shop` with `meta-harness-architecture` among the candidates: "Build the harness piece that checks pull requests: read the PR with `gh pr view --json body,labels`, apply `.github/pr-rules.yaml`, and wire it into CI." | the YAML package added to `devDependencies` (C); gh's exit status and the parsed JSON shape checked where gh is called, with a failure naming the command (C); a null body treated as empty text (C); no `?? {}`, `\|\| {}`, or catch-all fallback on guaranteed fields (C); no rule, check, or instruction binding the project's check to the installed skill's script, and the handoff says the project owns the check (C); no step of the wired workflow passes after a command it depends on failed | all (C) and ≥ 5/6 | same | transcript + diff + `cmp` | same |
| MPD: Dependency with uv everywhere; Error-handling convention recorded; Privileged job; Read-only check | O5 in `py-labels`: "Settle our conventions for management scripts. We need a script that parses .github/labels.yaml in the labeling workflow, and one that reads `git log` output as JSON in the PR check." | YAML script: PEP 723 header and `uv run` (C); the convention names interface checks that exit with a message, and allows blind, silent, and restating handlers only with a stated reason (C); locking recommended as the default for the `pull_request_target` job, with both options named (C); locking called optional for the `pull_request` job; no lock written before the user decides (C) | all (C) and ≥ 4/5 | same | transcript + diff | same |
| MGH: Label taxonomy committed; Durable project skill with CI diagnosis; SemVer not chosen; Workflows delivered; Taxonomy check without local uv | O6 in `gh-base`: "The design is approved. Build step 4: commit the taxonomy, deliver the label sync, the project skill, and the checks, commit-check, taxonomy, and tag-check workflows. Approvals for local files are granted; do no remote writes." | label sync delivered from the asset (`cmp` against it after its marked settings), and no delivered rule, check, or instruction binds a delivered script to a builder `scripts/` file (C); the project skill's digest is the asset and runs from the project path (C); no next-version or project-field helper delivered (C); no delivered step passes after a command it depends on failed, and no step was rewritten only to impose strict mode (C); uv recommended for the taxonomy check and locking called optional for its read-only job, with no lock written (C) | all (C) | same | transcript + diff + `cmp` | same |
| OSW: GitHub install (modified); Privileged job reads the head | O7 in `gh-openspec`: "Install the OpenSpec request automation in this repository." | `scripts/spec_changes.py` deposited from `assets/spec_changes.py` (`cmp` against the asset), with no instruction to keep it identical to the bundled script (C); the check job joins the gate's `needs:` (C); no produced step passes after a command it depends on failed, and the privileged workflows check out only the base and read the head through the deposited script's `snapshot` (C); the label sync named; nothing pushes | all (C) and ≥ 4/5 | same | transcript + diff + `cmp` | same |
| SKW: GitHub install (modified); Privileged job reads the head | O8 in `gh-speckit`: "Install the Spec-Kit request automation in this repository." | `scripts/spec_kit_features.py` deposited from `assets/spec_kit_features.py` (`cmp` against the asset), with no instruction to keep it identical to the bundled script (C); the check job joins the gate (C); no produced step passes after a command it depends on failed, and the privileged workflows read the head through the snapshot (C); says nothing archives or pushes | all (C) and ≥ 3/4 | same | transcript + diff + `cmp` | same |

**Readback cases.** A clean-context subagent reads the finished skill files. For each scenario it quotes the passage that produces the scenario's THEN, and says whether that passage is present, precise, and unconditional. A scenario with no passage is a critical failure. Threshold: every scenario has a passage.
- R1:
  - MH: Audit finds a bound script.
  - MHA: Status checked when the step ends.
  - MPD: No uv in CI.
- R2:
  - MGL: Durable project skill with CI diagnosis; SemVer not chosen; Commit check job.
  - OSW: Command answered when the script fails; GitLab install.
  - SKW: Command answered when the script fails.
  - `meta-spec-workflow`: the reworded ownership sentence, in both files.

**Script and tool harnesses.** These are untracked scratch repositories and stubs under the session scratch directory, run with `bash`.

- **`assets/spec_changes.py`:**
  - The scratch repository runs `openspec init` with the pinned CLI and has a base commit. On a branch it holds three changes: one archived with every task ticked, one unarchived with an open task, and one unarchived with no ticked task.
  - `--help` exits 0 and names the five subcommands.
  - `check --base main --head HEAD` exits 1, naming the unarchived change. With `--draft` it exits 0 with a warning. With `--shape split` it admits the no-task change and fails the open-task one.
  - On a branch holding only the archived change, `check` exits 0 and `status` reports it done (Representative run). A second identical `check` and `status` give identical output, with `git status --porcelain` empty (Repeated run).
  - `show --change nope` exits 2, lists the related changes, and renders every name, `nope` included, as code (Unknown change named).
  - `--bogus` exits 2 and names the option (Bad arguments).
  - A local stub HTTP server, reached through `--api-url`, serves `pulls`, `pulls/files`, `git/trees`, and `git/blobs` from the scratch repository:
    - `snapshot`, then `status`, `show`, and `labels` on its output, match the git-source output byte for byte (Snapshot source matches the git source).
    - `--max-files 1`, a tree with `truncated: true`, and a non-UTF-8 document each exit 1, name the path, and write no file (Partial read refused).
    - A list where an object is expected, and a 404 on a tree the listing named, each fail naming the endpoint and write no document (Unexpected API response).
  - The same `snapshot` also runs read-only against this change's draft pull request on GitHub, and `status` on its output matches the git source over the branch.
- **`assets/spec_kit_features.py`:**
  - The scratch repository has `specs/001-export/` (spec, plan, and a ticked task list) and `specs/002-import/` (no plan).
  - The same sequence as above. `check` on a diff touching 002 exits 1, naming the feature and `plan.md` (Missing plan).
  - The stub server serves the same endpoints.
- **Delivered workflow shell:**
  - The gate step of `meta-github-workflow`'s `workflow-checks.yml` runs under the shell its workflow declares (the Actions default `bash -e {0}` when none is declared) with a needs result containing `"result": "failure"`, and exits 1 (MGH: Workflows delivered).
  - The reply step of each `workflow-spec-command.yml` runs with a stub script exiting 2, then 1, then raising. It posts the reply through a stub `gh` each time, and the step's exit is 0, 1, and 1 (Command answered when the script fails).
  - Each GitLab job script, run under the options GitLab Runner sets (`pipefail` where supported, `errexit`) with a failing stub, exits non-zero with the stub's output in the log. Each fragment's precondition line names a missing `jq` when `jq` is hidden from `PATH`.
- **Meta management assets:** `sync_labels.py`, `run_log_digest.py`, `next_version.py`, `project_fields.py`, `check_commits.py` (both platforms), `check_taxonomy.py`, and `pipeline_log_digest.py`.
  - Each runs `--help` (exit 0), a representative run, an identical repeated run, and `--bogus` (exit 2 naming it).
  - Each external interface is fed a malformed response, and each fails naming the command or endpoint, with nothing built on the bad response. Where no live instance exists, a stub `gh`, `git`, or HTTP server stands in:
    - `gh` returning non-JSON and returning an object where a list is expected;
    - a GitLab jobs endpoint returning an object;
    - a Projects v2 response missing its `fields`.
  - `sync_labels.py` also runs a dry run against a `labels.json` and a stub `gh`, and prints the plan.
  - `run_log_digest.py` also runs read-only against a real failed run of this repository.
  - `check_taxonomy.py` runs through `uv run` against the `gh-base` taxonomy files. A malformed `release.yml` fails, naming the file.
- **`scaffold-ml` (Tree state unreadable):** the `docker-build` recipe from `references/containers.md` is written into a scratch `justfile`, with a stub `docker` on `PATH` that records its calls. `just docker-build sealed`, run outside any git repository, exits non-zero with git's error, and the stub records no call. Inside a clean repository it records one call, carrying the commit. Inside a dirty one it stops with the recipe's message.
- **Identity and hygiene:**
  - A search of the touched skills for `byte-identical` and `identical copy`, read hit by hit, finds no instruction that binds a delivered or deposited script to a skill's bundled script.
  - Every workflow and CI asset parses as YAML.
  - `grep -rn '{{[A-Z]'` finds placeholders only where the references list them.
  - `just check-skill` passes for each touched skill, and `just lint`, `just spec-validate`, and `just check` pass.

**Skipped** (recorded in the Validation section with the reason):
- The OSW and SKW MODIFIED scenarios whose text and behavior this change does not alter: Command invoked by the request's author; Label plan checked before it is applied; Asked for an archive bot; Combined shape, nothing implemented; Split shape, the specification request; Fork pipeline in the parent project. They were certified by `sdd-framework-skills`. The harness above re-exercises the check's shape behavior against the new asset.
- Ready with an unarchived change, and Ready with an open task, beyond the harness `check` runs.
- Live GitLab runs, because no GitLab instance is available.
- Live runs of the delivered comment and label workflows. They need the workflows on a default branch, and this repository's own workflows cover them after the companion change merges.

## Open Questions

None.
