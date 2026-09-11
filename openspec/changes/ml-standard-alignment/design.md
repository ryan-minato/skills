## Context

See proposal.md. `scaffold-ml` is a disposable topic builder in `scaffold`:
its description opens with the shared marker, it may depend by name on
`meta` builders and `core` skills (`skills/scaffold/CONTEXT.md`), and it
names the durable `machine-learning` skills only by role, routed through
`ryan-minato-skills-installing`, with a fallback the deposit carries (the
companion change adds that sentence to the catalog context). Today the
skill holds a 123-line SKILL.md with two modes, five references split by
mode (`references/{containers,experiment/pinned-deps,experiment/shared-module,training/hydra-config,training/uv-hardware-deps}.md`),
and two asset trees of six and seven files (Dockerfile, AGENTS.md,
compose, devcontainer, justfile, train.py, plus the training tree's
pyproject tool config). No spec domain exists yet. Limits: description
≤ 1024 characters (warn > 900) and it must keep the marker sentence;
body under 500 lines; no path outside the skill; no documentation-URL
index; assets are raw starting shapes reworked line by line. Mirrors
created by this change: `assets/run_manifest.py` is a copy of
`skills/machine-learning/experiment-provenance/assets/run_manifest.py`
(docstring first line may differ); `assets/openspec/research-task/templates/research.md`
carries the section headings of
`skills/machine-learning/research-workflow/assets/research-spec.md`; both
are registered by the companion change. The topic builder loads before
the `meta` entry and hands the rest of the harness to it; everything it
deposits is a settled choice for the builders that follow, so each rule
lands with its reason.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| Trigger: description | `SKILL.md` frontmatter `description` (marker kept) | — |
| Behavior: One project shape, with existing choices preserved | `SKILL.md` `## Inspect first` (goal, evaluation, data, hardware, environment, horizon, existing tools), `## The shape` (layout, framework default, entry points, data rule, outputs), `## Workflow` steps 1–4; `references/growing-the-layout.md`; `assets/train.py`, `assets/eval.py`, `assets/stages.py`, `assets/configs/config.yaml`, `assets/config.py`, `assets/justfile`, `assets/docs-data.md` | "Read `references/growing-the-layout.md` when two entry scripts need the same code or the project grows a package, tests, or a docs directory." |
| Behavior: The dependency carrier is the user's choice, a uv project by default | `SKILL.md` `## Workflow` step 3 (the carrier question with its recommendation); `references/hardware-deps.md` (uv project: indexes, sources, markers, multi-target extras); `references/requirements-lock.md` (four-file workflow, backend flag, per-machine backends); `assets/pyproject-tool-config.toml`, `assets/requirements.in`, `assets/requirements.dev.in`, `assets/justfile` (carrier-specific `setup`/`lock` recipes, one variant commented) | "Read `references/hardware-deps.md` when the carrier is a uv project and torch or another accelerator-bound package is added, or the hardware changes." / "Read `references/requirements-lock.md` when the carrier is the requirements workflow, when creating or updating the requirements files, or when a development machine differs from the training box." |
| Behavior: The configuration surface is typed values with a resolved dump | `SKILL.md` `## Workflow` step 4; `references/config-surface.md` (schema, YAML states, CLI overrides, resolved dump, exposure, existing Hydra); `assets/config.py`, `assets/configs/config.yaml`, `assets/train.py` (dump before the first step) | "Read `references/config-surface.md` when creating or restructuring `configs/`, when a configuration file starts naming classes or conditionals, or when the project already runs Hydra." |
| Behavior: Every run writes a manifest and logs to a selected tracker | `SKILL.md` `## Workflow` step 5; `references/provenance-and-tracker.md` (manifest fields, snapshot and retention rules, tracker precedence, `log_with` wiring, API verified at build time); `assets/run_manifest.py`, `assets/train.py` (start and finish calls, tracker through the logging seam), `assets/agents-md.md` `## Provenance` | "Read `references/provenance-and-tracker.md` when wiring the manifest and the tracker, or when the project already uses a tracker." |
| Behavior: The container recipe yields a recorded image identity | `SKILL.md` `## Workflow` step 8 (opt-in; handoff to the GPU container builder first); `references/containers.md` (three stages, base image and lock, preinstalled-stack rule, digest and `IMAGE_DIGEST`, host facts, volumes, shared memory, assets); `assets/Dockerfile`, `assets/compose.yaml`, `assets/devcontainer.json`, `assets/justfile` (`docker-build`, `docker-digest`) | "Read `references/containers.md` when the user asks for a dev container, a Compose environment, or a training image, after the GPU container builder's decisions." |
| Behavior: Tests, hooks, and style follow the experiment standard | `SKILL.md` `## Workflow` step 6 and `## Deposited decisions` (each with its reason); `assets/pyproject-tool-config.toml` (Ruff: line length 120, `docstring-code-format`, `docstring-code-line-length = 80`; pytest markers), `assets/pre-commit-config.yaml` (ruff only, comment on why), `assets/justfile` (`test`, `test-gpu`, `test-slow`, `lint`), `assets/agents-md.md` `## Code style`, `## Tests` | — |
| Behavior: The research-task convention is deposited for the project's spec tooling | `SKILL.md` `## Workflow` step 7; `references/research-task.md` (fields, one task per request, evolution, completion, spec location, OpenSpec schema placement verified from the tool's help, the unit-of-work note for the workflow builders); `assets/openspec/research-task/schema.yaml` and `templates/{research,hypotheses,tasks}.md`; `assets/research-spec.md` (no-tool skeleton); `assets/agents-md.md` `## Research tasks` | "Read `references/research-task.md` when depositing the research-task convention, when the project runs OpenSpec, or when the project runs no spec tool." |
| Behavior: The deposited guidance makes the project discoverable | `SKILL.md` `## Workflow` step 9; `assets/agents-md.md` (commands, configuration, tests, research tasks, tracker, provenance, error handling, a code-style section — abstraction rule, mature dependencies, explicit loop, block comments, hot-path performance, profiling on demand — the never-commit rule, when-to-read table naming the durable roles) | — |
| Behavior: The scaffold is verified before the handoff | `SKILL.md` `## Workflow` step 10 and `## Done when` | — |
| Handoff: GPU container environments | `SKILL.md` `## Workflow` step 8 (existing wording kept) | — |
| Handoff: work tracking and agent authority | `SKILL.md` `## Workflow` step 11 (existing wording kept) | — |
| Handoff: durable machine-learning skills | `SKILL.md` `## Workflow` step 12 (five roles, installer routing, fallback) | — |
| Handoff: harness entry and closing | `SKILL.md` `## Workflow` step 13 (existing wording kept: entry builder, else ask about deletion and load the disposal builder) | — |

Removed: `references/experiment/`, `references/training/`, `assets/experiment/`, `assets/training/` (their retained content moves to the files above; `pinned-deps.md` becomes `references/requirements-lock.md`, `uv-hardware-deps.md` becomes `references/hardware-deps.md`, `shared-module.md` becomes `references/growing-the-layout.md`, `hydra-config.md`'s two rules become the existing-Hydra section of `references/config-surface.md`).

## Description

Keeps the marker sentence verbatim, then: a capability sentence for one
reproducible ML project shape with provenance, a tracker, an image
identity, experiment-grade test and hook rules, a research-task
convention, and the durable skills by role; triggers for an empty or
early repository that trains or evaluates models ("set this repo up
properly", "turn this train.py into a project", "we need structure,
commands, checks, and agent guidance"); exclusions for a mature
migration, an inference-only application, and a data pipeline that
consumes models. Budget: under 900 characters with the marker.

## Dependencies and handoffs

- `meta-gpu-container`, `meta-workflow-design`, `meta-agent-authority`,
  `meta-harness-building`, `meta-disposal` (in range: `scaffold` may name
  `meta` builders), each with the whole-catalog installation through
  `ryan-minato-skills-installing` and the existing decline fallbacks.
- `ryan-minato-skills-installing` (`core`): the route for every handoff.
- The durable `machine-learning` skills: out of range; named by role in
  the description and the spec, by name in the body only as the example
  of the role with the installer routing; fallback: the deposited
  entrypoint rules stand alone with a note.
- No dependency on any other repository.

## External impact

- `skills/scaffold/README.md` and `README.zh.md`: the `scaffold-ml` row
  rewritten (content-identical pair); proof by reading both.
- `.claude-plugin/marketplace.json`: the `scaffold` plugin description
  loses "quick experiment or maintainable training codebase" wording if
  present; `skills[]` unchanged; proof `just gen-marketplace` then
  `git diff --exit-code` on the `skills` arrays and `just validate`.
- `skills/scaffold/CONTEXT.md` and `.agents/knowledge/harness-maintenance.md`:
  the companion change; proof there.
- `ryan-minato-skills-installing`'s install example naming `scaffold-ml`
  is unaffected (name unchanged).
- No `meta` file, no `machine-learning` file, no `scripts/` file changes;
  proof `git diff --stat origin/main...HEAD -- skills/meta skills/machine-learning skills/core skills/engineering skills/writing scripts` empty.

## Decisions

- **One shape, the carrier as a decision inside it** (serves One project
  shape; The dependency carrier): the removed fork was configuration and
  assets duplicated per mode; the carrier is orthogonal and costs one
  reference and two justfile recipes, so the user keeps the choice with a
  recommended default. Alternative rejected: uv project only, which the
  maintainer declined.
- **OmegaConf structured schema, not Hydra, as the default** (serves The
  configuration surface): the standard rejects composition frameworks as
  the default; Hydra stays for projects that already run it with the two
  rules. `assets/config.py` holds the dataclasses so the schema is code,
  not YAML.
- **The manifest module is a byte-level copy of the durable skill's**
  (serves Every run writes a manifest): one implementation, two homes,
  registered as a mirror; the scaffold's copy is what a project without
  the durable skill still gets.
- **The research-task schema ships as an asset here, not in `meta`**
  (serves The research-task convention): the maintainer's rule that
  `meta` stays generic; the schema follows this repository's own
  `openspec/schemas/skill-change/schema.yaml` shape (three artifacts, no
  specs artifact, `skip_specs` changes) and is placed at the tool's
  documented schema path, with the tool's schema commands verified from
  `openspec --help` at build time rather than quoted.
- **Deposited decisions carry reasons** (serves Tests, hooks, and style;
  The deposited guidance): the `meta` builders inventory settled choices
  and never migrate a working one, so a reason beside each rule is what
  keeps them from re-proposing their baseline.
- **Verification runs the deposited commands, not a validator script**
  (serves The scaffold is verified): a validator like the data-science
  scaffold's is a later change if the scaffold earns one; the smoke run
  on a tiny input is what proves the loop, the manifest, and the
  evaluation entry work together.
- **Existing GPU-container, workflow, authority, entry, and disposal
  handoff wording is kept verbatim** so the four handoffs stay
  consistent with the sibling scaffolds.

## Risks / Trade-offs

- [SKILL.md grows past the 500-line warning with thirteen steps] → the
  workflow stays terse; every branch (carrier, containers, research
  tooling, layout growth, configuration, provenance) lives in a reference
  with a precise load sentence.
- [Assets drift from the durable skills' copies] → the companion change
  registers both mirrors with a `diff` proof; the manifest copy differs
  only in its docstring's first line.
- [The `meta` builders re-propose their baseline over the scaffold's
  deposit] → each deposited decision states its reason and is placed
  where those builders inventory (the entrypoint and the knowledge
  file), which their contracts treat as settled; the spec's
  "runs afterwards" scenarios are read back against the deposited text.
- [OpenSpec's schema commands are experimental in the pinned version] →
  the reference names the placement path and says to verify the command
  set from the tool's help; the fallback scenario copies files by hand
  and records the version.
- [Tracker APIs move] → the reference names capabilities and the seam,
  never method names; the wiring is verified at build time.
- [A BREAKING change to a builder installed elsewhere] → installed copies
  are never updated in place; the commit is `feat(scaffold-ml)!` and the
  proposal marks it.
- [Test budget] → one load and one near-miss Trigger prompt run as
  solvers; every other scenario is read back by a clean-context subagent
  against the finished skill; the assets are smoke-tested in a scratch
  environment.

## Verification plan

Solver tier: Sonnet-class. Observation: the appended neutral
`SKILLS_LOADED:` self-report. Isolation: fresh clean-context subagent per
prompt, a throwaway fixture repository under the session scratch
directory (an empty repository with a README saying it will fine-tune a
small language model), the builder visible through this repository's
`.agents/skills` symlink; one attempt per case, up to three on an invalid
observation.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| Trigger: Empty training repository | the scenario's prompt, verbatim, in the fixture | loads `scaffold-ml` (critical) | 1/1 | Sonnet-class | as above | as above |
| Trigger: Data pipeline (near-miss) | the scenario's prompt, verbatim | does not load `scaffold-ml` (critical) | 1/1 | same | same | same |

Readback cases (one clean-context subagent reads the finished skill
directory and the delta spec, quotes the passage behind every remaining
scenario's THEN, and reports PRESENT / WEAK / GAP; a GAP is a critical
failure; threshold: no GAP): Early repository with one script; Inference
service (near-miss); every Behavior scenario of the nine Behavior
requirements; Handoff offered and User declines for the four handoffs.
The readback also checks the marker sentence, link containment, the
installer pattern on every handoff, the absence of `{{` placeholders in
assets that the SKILL.md does not order reworked, and the load sentence
of every reference.

Asset harness (scratch directory outside version control):
- `python3 -m py_compile` on every `.py` asset; `ruff check` and
  `ruff format --check` on them.
- `diff` of `assets/run_manifest.py` against
  `skills/machine-learning/experiment-provenance/assets/run_manifest.py`
  shows only the docstring's first line.
- A CPU smoke run in a scratch venv with torch, accelerate, and
  omegaconf installed: copy `train.py`, `eval.py`, `config.py`,
  `stages.py`, `run_manifest.py`, and `configs/config.yaml` into a
  fixture, run `python train.py steps=3` and `python eval.py`, and check
  that `outputs/<run_id>/config.resolved.yaml`, `manifest.json`, and
  `manifest.running.json` exist and that the manifest's `config.sha256`
  matches the dump. Recorded as skipped with the reason if the packages
  cannot be installed.
- `python3 -c "import yaml; yaml.safe_load(open('assets/openspec/research-task/schema.yaml'))"`
  and `openspec validate --strict` on a scratch change created from the
  schema after copying it into a scratch OpenSpec project, when the
  pinned CLI accepts a project-local schema; otherwise recorded.
- `just check-skill skills/scaffold/scaffold-ml`, `just lint`,
  `just spec-validate`, `just check`.

Skipped (recorded in the Validation section with this reason): every
Trigger scenario not in the table above and every Behavior and Handoff
scenario is not executed by a solver — the maintainer limited the fleet
to one load and one near-miss prompt; each is covered by the readback.

## Open Questions

None.
