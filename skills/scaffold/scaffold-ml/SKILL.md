---
name: scaffold-ml
description: >-
  Disposable builder skill (delete after the harness is built): scaffolds
  one reproducible machine-learning project shape and its agent harness —
  a typed configuration surface with a resolved dump per run, a dependency
  carrier the user chooses (uv project by default), an explicit Accelerate
  loop with a run manifest and a tracker, an opt-in container recipe
  whose image digest is the environment identity, experiment-grade test
  and hook rules, a research-task convention, and pointers to the durable
  machine-learning skills by role. Use for an empty or early repository that trains or
  evaluates models: "set this repo up properly", "turn this train.py and
  notebook into a project", "we need structure, commands, checks, and agent
  guidance before the team grows". Not for a mature project's migration,
  an inference-only application, or a data pipeline that consumes models
  without training them.
license: Apache-2.0
---

# Machine-Learning Project Scaffold

Build only after inspecting the project. Precedence on every decision: an
explicit user instruction, then a working choice the repository already
makes, then this builder's defaults, then tool preferences. Scaffolding is
not permission to migrate. Everything deposited here is read afterwards
by the generic harness builders as a settled project choice, so every
rule lands with its reason.

## Inspect first

Record before writing: what is trained and evaluated and what counts as
"better" (the benchmark or evaluation set and its metric definitions —
ask when nothing states it; evaluation is not a later task); the data
sources, their identities, scale, and location; the target hardware and
accelerator stack and whether development machines differ from the
training box; local versus remote execution; the maintenance horizon; the
hosting platform (it selects the tracker); every working tool, lock,
configuration system, hook, or structure already present.

## The shape

One shape for every project, sized by the extraction rule rather than by
a mode:

```text
train.py, eval.py            thin entry points (explicit loop; evaluation bound to the benchmark)
config.py, configs/*.yaml    typed schema + named states; overrides on the command line
<project_name>/              shared code, created only when two places must stay consistent
data/raw/                    local cache of immutable inputs; no transformation writes here
outputs/<run_id>/            config.resolved.yaml, manifest.json, checkpoints, exports; never committed
tests/                       behavior contracts; `gpu` and `slow` markers
research/<task>/             the research spec and hypothesis log (or the spec tool's change directory)
AGENTS.md, docs/data.md      the harness: commands, rules, when-to-read
```

- Framework: PyTorch with Accelerate by default; a framework the project
  already uses stays. When the inventory shows a JAX signal — TPU, a
  differentiable simulation or solver, scientific or numerical research
  where the computation outweighs the model, higher-order derivatives,
  heavy `grad`/`vmap`/`jit` composition, large homogeneous parallel
  computation, compiler or autodiff research — evaluate JAX and give the
  decision to the user. Read [references/framework-choice.md](references/framework-choice.md)
  before fixing the framework of an unsettled project or when any of
  those signals appears. A JAX project keeps only the framework-neutral
  assets: both entry points (`train.py`, `eval.py`) are PyTorch-only and
  are written from that reference's loop shape, and the dependency
  declaration and base image follow the JAX sections of
  `references/hardware-deps.md` and `references/containers.md`.
- The training entry point is an explicit loop on Accelerate: seeding,
  gradient accumulation, clipping, checkpoint save and resume, an
  evaluation cadence, one logging seam, one stage-trace seam; multi-device
  runs launch through the launcher, never by editing the loop.
- Read [references/growing-the-layout.md](references/growing-the-layout.md)
  when two entry scripts need the same code or the project grows a
  package, tests, or a docs directory.

## Workflow

1. **Inventory** (above). Present the concrete project shape you intend
   to create — files, commands, the decisions below — before creating it.
   Before the first commit, add every skill directory whose description
   opens with `Disposable builder skill (delete after the harness is built):`
   to `$(git rev-parse --git-path info/exclude)`; stage explicit paths and
   read `git status` before each commit — the builders never enter one.
2. **Framework and evaluation.** Fix the framework: the default without
   a question, a JAX evaluation with a recommendation and the user's
   decision when a signal is present, the existing framework without any
   evaluation otherwise; record it with its deciding signal in
   `AGENTS.md`. Record the
   benchmark or evaluation set identity and the metric definitions;
   `eval.py` is bound to them from the first run and records each
   evaluation as a child run of the training run.
3. **Dependency carrier — the user's choice.** Recommend a uv project
   (`pyproject.toml` with a `dev` dependency group, `uv.lock`, accelerator
   wheels routed through explicit indexes and sources) and ask the user
   to confirm it or choose the requirements workflow (`requirements.in`
   and `requirements.dev.in`, each compiled by uv into a fully pinned
   `.txt` with the torch backend flag; the training box syncs the runtime
   file, a development machine the dev file). Keep a carrier the
   repository already uses without asking. In either carrier the runtime
   lock is the environment identity; a requirements file with ranges is
   not a lock. Read [references/hardware-deps.md](references/hardware-deps.md)
   when the carrier is a uv project and torch or another
   accelerator-bound package is added, or the hardware changes. Read
   [references/requirements-lock.md](references/requirements-lock.md)
   when the carrier is the requirements workflow, when creating or
   updating the requirements files, or when a development machine differs
   from the training box. Copy [`justfile`](assets/justfile); for the
   requirements carrier replace its `setup`, `setup-train`, and `lock`
   recipes with the block in that reference and copy
   [`requirements.in`](assets/requirements.in) and
   [`requirements.dev.in`](assets/requirements.dev.in), filling the
   model and data libraries.
4. **Configuration surface.** Default: a dataclass schema, named states
   in `configs/*.yaml`, command-line overrides, and one resolved document
   written to `outputs/<run_id>/config.resolved.yaml` before the first
   step. Copy [`config.py`](assets/config.py) and
   [`configs/config.yaml`](assets/configs/config.yaml) and fill every
   placeholder and every `MISSING` value; add the fields the project's
   data, model, and optimizer take. The schema keeps the loop's
   control-plane fields and the named choices (`model.name`,
   `optim.name`); construction stays in `train.py`. Expose
   values a run may choose — named choices, never import paths,
   registries, control flow, or deep inheritance; call searched values
   that are not hyperparameters search variables. Read
   [references/config-surface.md](references/config-surface.md) when
   creating or restructuring `configs/`, when a configuration file starts
   naming classes or conditionals, or when the project already runs Hydra.
5. **Manifest, loop, and tracker.** Copy
   [`run_manifest.py`](assets/run_manifest.py) and
   [`stages.py`](assets/stages.py) unchanged. Copy
   [`train.py`](assets/train.py) and fill its placeholders — model,
   optimizer, training loader, scheduler, forward pass, evaluation,
   resume step, parent run — without touching its seams: it writes the
   manifest at start and finalizes it at the end (the start-time record
   is kept), and metrics leave only through `log_metrics`. Copy
   [`eval.py`](assets/eval.py) and fill its model-load and
   evaluation-loader placeholders; it imports `evaluate` from
   `train.py`. Select
   the tracker by precedence: a working tracker the project already uses;
   else the hosting platform's experiment tracking when it provides one
   (GitLab's experiments); else Trackio. Wire it through the loop's single
   logging seam; the manifest starts before the tracker so its identity
   fields ride along as the tracker's parameters; once it is wired, set
   `run.tracker` in `configs/config.yaml`. Deposit the snapshot
   rule (commit before every run; the entry points refuse a dirty tree,
   and `run.allow_dirty=true` is the one override, for a throwaway run
   marked degraded) and the retention rule (a tag per run or a kept
   research branch keeps cited snapshots reachable after a squash). Read [references/provenance-and-tracker.md](references/provenance-and-tracker.md)
   when wiring the manifest and the tracker, or when the project already
   uses a tracker.
6. **Tests, hooks, style — each with its reason.** pytest markers `slow`
   and `gpu`; `just test` is light and CPU-compatible; `just test-gpu`
   runs the GPU suite and fails without hardware (a silent skip hides a
   broken test — when asked to make them skip, keep them failing under
   `just test-gpu`, out of the default suite, and say why); `just
   test-slow` is manual. Ruff near its defaults (line
   length 120, docstring code formatting, docstring code line length 80).
   No static type checker as a gate over model code — offer it for
   `config.py` and control-plane modules only; docstrings carry shape,
   dtype, device, and mask conventions. Git hooks run the formatter and
   linter only: commits are experiment snapshots and a hook that runs
   tests taxes every snapshot; an existing hook that runs tests gets a
   proposal to limit it, with that reason, and the user's decision
   stands. Merge
   [`pyproject-tool-config.toml`](assets/pyproject-tool-config.toml)
   into `pyproject.toml` — a uv project also declares the runtime
   dependencies under `[project]` and the index routing from
   `references/hardware-deps.md`; the requirements carrier drops
   `[dependency-groups]`. Copy
   [`pre-commit-config.yaml`](assets/pre-commit-config.yaml) to
   `.pre-commit-config.yaml` and pin `rev` to the current
   ruff-pre-commit release.
7. **Research-task convention.** Deposit the research spec's fields
   (Objective and Evaluation always; Context, Search Scope, Constraints,
   Completion Condition, Hypotheses as needed), one research task per
   pull or merge request carrying its hypotheses and runs, the spec
   evolving while run history stays immutable, and completion on the
   completion condition with negative results as valid outcomes. Place
   the spec inside the project's specification contract when one exists.
   When the project runs OpenSpec, copy
   [`assets/openspec/research-task/`](assets/openspec/research-task/schema.yaml)
   whole to the tool's schema directory and record how a change selects
   it; with no spec tool, copy its
   [`templates/research.md`](assets/openspec/research-task/templates/research.md)
   to `research/<task>/spec.md` and
   [`templates/hypotheses.md`](assets/openspec/research-task/templates/hypotheses.md)
   beside it.
   Record the research task as the project's unit of research work. Read
   [references/research-task.md](references/research-task.md) when
   depositing the convention, when the project runs OpenSpec, or when it
   runs no spec tool.
8. **Containers — opt-in.** Only when the user asks for a dev container,
   a Compose environment, or a training image. Make the generic decisions
   — whether a container is warranted, base image and live tag, GPU
   wiring — with the `meta-gpu-container` skill first. If it is not
   installed, load the `ryan-minato-skills-installing` skill and install
   the whole `meta` catalog at project scope as it directs, never
   `meta-gpu-container` alone — its builders stack, and `meta-disposal`
   removes both catalogs' builders in the closing step. Never run an
   install command yourself. If the user declines, proceed with the
   reference alone and record in the handoff that the base image was
   chosen without the GPU container builder's live verification. Then
   read
   [references/containers.md](references/containers.md) — the three-stage
   recipe, the digest as the environment identity, the environment outside
   the source mount, the build-context filter, volumes, shared memory.
   Copy [`Dockerfile`](assets/Dockerfile) to the project root and pin the
   uv image tag and the live-enumerated base tag; copy
   [`compose.yaml`](assets/compose.yaml) and fill `shm_size`; copy
   [`devcontainer.json`](assets/devcontainer.json) to
   `.devcontainer/devcontainer.json`, fill the image and the
   shared-memory size, and insert the GPU flags the container decision
   produced into `runArgs` as separate items (NVIDIA `"--gpus", "all"`;
   ROCm `"--device", "/dev/kfd", "--device", "/dev/dri"`); copy
   [`dockerignore`](assets/dockerignore) to `.dockerignore`; append the
   task-runner recipes from that reference to the justfile.
9. **Deposit the guidance.** Merge [`gitignore`](assets/gitignore) into
   `.gitignore` (an unignored `outputs/` or `data/` makes every run's
   tree dirty, and the entry points refuse a dirty tree). Rework
   [`agents-md.md`](assets/agents-md.md) into `AGENTS.md`: fill every
   `<…>` slot and keep every section — together they are the guidance
   the specification requires — with one authoritative source per fact.
   Copy [`docs-data.md`](assets/docs-data.md) to `docs/data.md` and fill
   the source rows and the evaluation line. No generated file carries
   this builder's marker.
10. **Verify.** Run `just setup`, `just lint`, `just test`, and a smoke
    run of `just train` and `just eval` on a tiny input; confirm every
    link in `AGENTS.md` resolves and `grep -rn "Disposable builder"` over
    the project finds nothing; inspect the result with the user. A
    command that cannot run on this machine (no accelerator, no network)
    is reported as not run with the command to run it elsewhere — never
    skipped silently.
11. **Work tracking and authority.** Work tracking, planning, and
    agent-autonomy rules are designed with the `meta-workflow-design` and
    `meta-agent-authority` skills, not improvised here — do not invent an
    issue, review, or autonomy flow in the scaffold. If they are not
    installed, load the `ryan-minato-skills-installing` skill and install
    the whole `meta` catalog at project scope as it directs (one install
    covers every `meta` builder this workflow names); never run an install
    command yourself. If the user declines, leave management design out
    and record the gap in the handoff.
12. **Durable machine-learning skills.** The project's ongoing guidance
    lives in five durable roles — run provenance, the research task,
    experiment code conventions, training instrumentation, training
    diagnosis — published as the `machine-learning` catalog
    (`experiment-provenance`, `research-workflow`,
    `experiment-code-conventions`, `training-instrumentation`,
    `training-diagnosis`). Load the `ryan-minato-skills-installing` skill
    and install them as it directs; never run an install command
    yourself. (If that installer skill is absent too, it lives in the
    `core` catalog of https://github.com/ryan-minato/skills.) Record in
    the when-to-read table where each role applies. If the user declines,
    the deposited `AGENTS.md` rules — manifest, snapshot, tracker, test
    markers, hooks — stand alone; note that the durable skills are not
    installed.
13. **Hand off and close.** Hand the rest of the harness — entrypoint
    depth, knowledge, project skills, synchronization — and the closing
    of the build to the `meta-harness-building` skill; when it is not
    installed, load the `ryan-minato-skills-installing` skill and install
    the whole `meta` catalog at project scope as it directs, never running
    an install command yourself. If the user declines that handoff, close
    here: once the deposit is
    verified and before the work goes to review, ask the user whether to
    delete the disposable builders now — the build request is not
    deletion consent — and on that decision load `meta-disposal`, which
    lists, confirms, and removes them. If they keep the builders, leave
    them in place and out of every commit, and record it in the handoff.

Done when: a fresh checkout on the stated hardware can set up and run
the documented train and eval commands; a run leaves
`config.resolved.yaml`, `manifest.json`, and `manifest.running.json`
under `outputs/<run_id>/`; the tracker receives the manifest fields;
fast checks pass; every `AGENTS.md` link resolves; no placeholder and no
marker remains; and the user has approved the shape.

## Deposited decisions and their reasons

| Decision | Reason the later builders keep it |
|---|---|
| No type-check gate over model code | shape, dtype, device, and mask contracts are not what a type checker checks; forcing it clean costs casts without credibility |
| Hooks run formatter and linter only | commits are experiment snapshots; tests belong to the task runner and CI |
| GPU tests fail without hardware | a skip makes the suite green on every machine that cannot run it |
| Runtime lock is the environment identity | a ranges file resolves differently on two days |
| Research task is the unit of research work | hypotheses and runs are its contents, not work items |
| `data/raw/` is immutable | a transformed cache silently corrupts every later comparison |

## Gotchas

- `run.mixed_precision=no` on the command line parses as YAML `false`;
  `load_config` maps it back before the resolved dump — keep that
  normalization when reworking `train.py`.
- `{{args}}` and `{{run_dir}}` in the justfile are just's recipe syntax,
  not placeholders to remove.
