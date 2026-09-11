## Why

The `machine-learning` catalog (#85) now carries the durable skills for
ML experiment projects, but the builder that initializes such a project
still describes a different standard: `scaffold-ml` scaffolds two shapes
(a quick experiment on Pydantic Settings and requirements files, a
training project on Hydra) with no run provenance, no tracker, no
resolved-configuration dump, no recorded image identity, no research-task
convention, and none of the test, hook, and typing deviations tensor code
needs. A project built from it today receives a harness that contradicts
the skills it will install tomorrow. The `meta` builders stay the generic
base: adapting that base to a machine-learning project is the topic
builder's job, deposited before the `meta` builders run so they find the
decisions settled. Now, before the next ML project is scaffolded.

## What Changes

- `scaffold-ml` — **BREAKING**: one project shape replaces the quick
  experiment / maintainable training choice: OmegaConf typed schema plus
  YAML plus command-line overrides with a resolved dump saved per run, a
  dependency carrier the user chooses — a uv project (`pyproject.toml`
  with a `dev` dependency group plus `uv.lock` with index routing for
  accelerator wheels) by default, or the four-file requirements workflow
  (`requirements.in` and `requirements.dev.in`, the latter including the
  former plus the development tools, each compiled by uv into a fully
  pinned `.txt` with the torch backend flag; the training machine syncs
  the runtime file, a development machine the dev file) for a
  scripts-only repository — in either carrier the runtime lock is the
  environment identity — an explicit Accelerate loop with one logging seam and one
  stage-trace seam, a run manifest written at start and finish, a tracker
  selected by precedence (keep existing → the hosting platform's
  experiment tracking → Trackio), a multi-stage container recipe whose
  pushed digest is the environment identity with the host facts a
  container cannot pin recorded beside it (containers stay opt-in),
  pytest `slow` and `gpu` markers with GPU tests that fail without
  hardware, Ruff at line 120 with docstring code formatting, no static
  type gate over model code, hooks that never run tests, a research-task
  convention (the spec's fields, one task per pull or merge request, the
  spec location, and — when the project's spec tool is OpenSpec — a
  project-local `research-task` schema deposited from an asset), and an
  AGENTS.md deposit that points at the research-spec location, the
  tracker, the configuration directory, the run-record convention, and —
  when installed — the durable machine-learning skills, named by role.
  Retained from the current builder and carried into the single shape:
  PyTorch as the default framework (JAX only on request or by ecosystem),
  the loop template's seeding, accumulation, checkpoint save and resume,
  and multi-device launch, an evaluation entry point bound to a recorded
  benchmark identity, the immutable `data/raw/` rule, the hardware-deps
  routing for development machines that differ from the training box,
  the container rules for preinstalled-stack images, volumes, and shared
  memory, the let-it-crash error convention, and the verification run
  before handoff. Every deposited decision carries its reason so the `meta` builders that
  run afterwards (Python defaults, workflow design, specification
  workflow, GPU containers, platform lifecycle) keep it as a settled
  project choice. Existing Hydra, Pydantic Settings, or requirements-file
  projects are preserved under the existing-choice rule (Hydra: no
  object instantiation, a pinned output directory). The Hydra reference
  and both asset trees are removed; the requirements-compile reference
  stays as the carrier's branch; the description no longer offers two
  modes.
- No `meta` builder changes: the generic base stays generic.

## Skills touched

- `scaffold/scaffold-ml` (new): description triggers, the single shape,
  the dependency carrier, the configuration surface, provenance and
  tracker, the image identity, tests and hooks, the research-task
  convention, the deposited guidance that later builders keep, and the
  handoffs to the GPU container builder, the workflow and authority
  builders, the harness entry, and the durable machine-learning skills by
  role.

## Installed behavior

`scaffold-ml`: a project is scaffolded in one shape with provenance, a
tracker, an image identity, the marker, typing, and hook rules, a
research-task convention, and pointers to the durable skills; the
two-mode choice and Hydra by default are gone, and the dependency carrier
is the user's explicit choice → `feat!`.

## Impact

- README pair rows: the `scaffold` row for `scaffold-ml` (both
  languages) drops the two-mode wording and names the new capabilities.
- `.claude-plugin/marketplace.json`: the `scaffold` plugin description if
  its wording moves (a human-owned field); the `skills[]` list is
  unchanged (no directory added or removed).
- `skills/scaffold/CONTEXT.md`: one sentence that durable topic skills of
  another catalog are outside the range and are named by role with a
  deposited fallback — the companion repository change.
- `.agents/knowledge/harness-maintenance.md`: mirror rows for the run
  manifest module (`experiment-provenance/assets/run_manifest.py` ↔
  `scaffold-ml/assets/run_manifest.py`) and the research-spec section
  headings (`research-workflow/assets/research-spec.md` ↔ the
  `research-task` schema template in `scaffold-ml/assets/`) — the
  companion repository change.
- Mirrored files in `scripts/validate_harness.py`: none. Symlinks:
  unchanged. No `meta` skill, reference, or asset changes.

## Non-goals

- Any change to a `meta` builder. In particular: the platform builders'
  ML references keep their current run-record shape (in an ML project
  the scaffold's deposited run-record convention is the settled choice
  the platform builder implements); `meta-spec-workflow` learns no
  research paradigm (the scaffold deposits the research-task convention
  and, for OpenSpec projects, the schema; the spec workflow builder keeps
  a spec tool and schema the project already runs);
  `meta-python-defaults` learns no tensor branch (the scaffold deposits
  the typing, test, and hook decisions with their reasons, which that
  builder inventories as settled); `meta-gpu-container` learns no digest
  rule (the scaffold's container reference carries it);
  `meta-workflow-design` gains no research-task sentence (the scaffold
  records the research task as the project's unit of research work in
  its knowledge, which that builder reads during inspection).
- Any change to the five `machine-learning` skills.
- A validator script for the scaffold; a later change if the scaffold
  earns one.
- Changing `scaffold-data-science`'s `data/<source>/` rule or reconciling
  it with the ML scaffold's `data/raw/`.

## Tracked work

No issue: the follow-up planned in conversation when #85 was proposed.
