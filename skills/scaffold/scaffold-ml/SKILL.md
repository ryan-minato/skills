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

- Framework: PyTorch for an otherwise empty project; JAX only when the
  user asks or the project already depends on that ecosystem.
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
2. **Framework and evaluation.** Fix the framework. Record the benchmark
   or evaluation set identity and the metric definitions; `eval.py` is
   bound to them from the first run.
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
   from the training box.
4. **Configuration surface.** Default: dataclass schema in
   [`config.py`](assets/config.py), named states in
   [`configs/config.yaml`](assets/configs/config.yaml), command-line
   overrides, and one resolved document written to
   `outputs/<run_id>/config.resolved.yaml` before the first step. Expose
   values a run may choose — named choices, never import paths,
   registries, control flow, or deep inheritance; call searched values
   that are not hyperparameters search variables. Read
   [references/config-surface.md](references/config-surface.md) when
   creating or restructuring `configs/`, when a configuration file starts
   naming classes or conditionals, or when the project already runs Hydra.
5. **Manifest and tracker.** Copy [`run_manifest.py`](assets/run_manifest.py)
   unchanged; [`train.py`](assets/train.py) writes the manifest at start
   and finalizes it at the end (the start-time record is kept). Select
   the tracker by precedence: a working tracker the project already uses;
   else the hosting platform's experiment tracking when it provides one
   (GitLab's experiments); else Trackio. Wire it through the loop's single
   logging seam. Deposit the snapshot rule (commit before every run; a
   dirty tree launches nothing) and the retention rule (a tag per run or
   a kept research branch keeps cited snapshots reachable after a
   squash). Read [references/provenance-and-tracker.md](references/provenance-and-tracker.md)
   when wiring the manifest and the tracker, or when the project already
   uses a tracker.
6. **Tests, hooks, style — each with its reason.** pytest markers `slow`
   and `gpu`; `just test` is light and CPU-compatible; `just test-gpu`
   runs the GPU suite and fails without hardware (a silent skip hides a
   broken test); `just test-slow` is manual. Ruff near its defaults (line
   length 120, docstring code formatting, docstring code line length 80).
   No static type checker as a gate over model code — offer it for
   `config.py` and control-plane modules only; docstrings carry shape,
   dtype, device, and mask conventions. Git hooks run the formatter and
   linter only: commits are experiment snapshots and a hook that runs
   tests taxes every snapshot. Copy
   [`pyproject-tool-config.toml`](assets/pyproject-tool-config.toml) and
   [`pre-commit-config.yaml`](assets/pre-commit-config.yaml).
7. **Research-task convention.** Deposit the research spec's fields
   (Objective and Evaluation always; Context, Search Scope, Constraints,
   Completion Condition, Hypotheses as needed), one research task per
   pull or merge request carrying its hypotheses and runs, the spec
   evolving while run history stays immutable, and completion on the
   completion condition with negative results as valid outcomes. Place
   the spec inside the project's specification contract when one exists,
   else under `research/<task>/` from [`research-spec.md`](assets/research-spec.md).
   When the project runs OpenSpec, copy
   [`assets/openspec/research-task/`](assets/openspec/research-task/schema.yaml)
   to the tool's schema directory and record how a change selects it.
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
   reference alone. Then read
   [references/containers.md](references/containers.md) — the three-stage
   recipe, the digest as the environment identity, volumes, shared memory.
9. **Deposit the guidance.** Rework [`agents-md.md`](assets/agents-md.md)
   into `AGENTS.md` and [`docs-data.md`](assets/docs-data.md) into
   `docs/data.md`: the commands, where configuration enters and how it is
   overridden, the test suites, the research-spec location, the tracker,
   the provenance rules, the error-handling convention (catch only
   expected data or environment failures; fail early with context for
   everything else), the code-style section, the never-commit rule for
   data, weights, checkpoints, and credentials, and the when-to-read
   table. One authoritative source per fact. No generated file carries
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
    installed, the same `meta` catalog install from step 8 covers them; if
    the user declines, leave management design out and record the gap in
    the handoff.
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
    the deposited `AGENTS.md` rules stand alone; note that the durable
    skills are not installed.
13. **Hand off and close.** Hand the rest of the harness — entrypoint
    depth, knowledge, project skills, synchronization — and the closing
    of the build to the `meta-harness-building` skill; the same `meta`
    catalog install from step 8 covers it when it is not installed. If
    the user declines that handoff, close here: once the deposit is
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

- Disposable builders never enter a commit: before the first commit, add
  every skill directory whose description opens with
  `Disposable builder skill (delete after the harness is built):` to
  `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and
  read `git status` before each commit.
- A run launched from a dirty tree records a commit that is not the code
  that ran; the manifest marks it degraded, and the guidance forbids it.
- `latest`, a branch, a Dockerfile, and a tag are names, not identities:
  record revisions, checksums, and digests.
- A preinstalled-framework image and a locked environment conflict: one
  of them is authoritative, and the guidance says which.
- Readability beats abstraction in experiment code; the extraction test
  is "must these stay consistent?", never "do these look alike?".
