---
name: scaffold-ml
description: >-
  Disposable scaffold skill (delete after the harness is built): scaffolds an
  opinionated, reproducible machine-learning project and its agent harness,
  choosing between a quick experiment and a maintainable training codebase.
  Use for empty or early
  repositories that train or evaluate models and need structure, hardware-aware
  dependencies, commands, checks, optional containers, and durable agent
  guidance. Not for mature migrations, inference-only applications, or
  replacing working project choices.
license: Apache-2.0
---

# Machine-Learning Project Scaffold

Build only after inspecting the training goal, evaluation contract, data
sources and scale, target hardware, execution environment, maintenance horizon,
and every working tool or structure already present.

## Choose the project mode

- **Quick experiment**: one idea, short lifetime, flat and readable layout,
  minimal configuration, and the shortest reproducible train/eval loop. Use the
  [experiment references](references/experiment/) and
  [experiment assets](assets/experiment/) only in this mode.
- **Maintainable training project**: months of extension, multiple workflows,
  reusable components, hardware-aware dependencies, structured configuration,
  and stronger checks. Use the [training references](references/training/) and
  [training assets](assets/training/) only in this mode.

Do not blend both shapes. Preserve an existing coherent mode unless the user
explicitly asks for migration.

## Shared workflow

1. Confirm what is trained and evaluated, hardware and accelerator stack, data
   identity and location, local versus remote execution, reproducibility needs,
   and expected maintenance horizon.
2. Choose the framework from project evidence. PyTorch is the default for an
   otherwise empty conventional training project; use JAX only when requested
   or when the project already depends on that ecosystem.
3. Establish a committed lockfile, immutable source-data rule, explicit output
   locations, seed and revision recording, and thin workflow entrypoints.
4. Apply the selected mode's layout, training-loop, configuration, command,
   check, and AGENTS.md assets. Rework every line; unresolved placeholders are
   failed decisions.
5. Add only feedback whose cost the lifecycle earns: fast lint/format and
   contract tests by default, with GPU, full-training, large-data, and
   equivalence tests behind explicit slow commands.
6. Containers are opt-in. When the user requests a dev container, Compose
   environment, or remote training image, make the generic GPU container
   decisions — whether a container is warranted, CUDA/ROCm base-image
   selection with live tag discovery, and GPU access wiring — with the
   `meta-gpu-container` skill, then load the selected mode's `containers.md`
   reference for the ML-specific integration. If `meta-gpu-container` is not
   installed, install it from https://github.com/ryan-minato/skills.git:

       npx skills add ryan-minato/skills --skill meta-gpu-container

7. Deposit durable project rules and when-to-read pointers into the target
   harness. Do not copy the disposable marker into any generated file.
8. Run the selected commands and inspect the result with the user.

## Mode requirements

### Quick experiment

- Keep the repository flat and obvious; extract shared code only when two
  places must remain logically consistent.
- Prefer one train entrypoint and one evaluation path.
- Make setup-to-first-run short, but still record hardware, dependency lock,
  data/model revisions, seeds, and outputs.
- Avoid architecture or configuration machinery meant for hypothetical future
  growth.

Done when: a fresh checkout on the stated hardware can set up and run the
documented experiment command, and the result records enough identity to be
reproduced.

### Maintainable training project

- Separate reusable package code, train/eval workflows, configuration, raw and
  processed data boundaries, outputs, focused tests, and optional notebooks.
- Expose values expected to vary; do not turn configuration into arbitrary
  class-path assembly.
- Route hardware-bound dependencies before adding them; a default package index
  may install the wrong accelerator build.
- Keep slow training and GPU tests out of pre-commit and default CI.

Done when: a fresh checkout on the stated hardware can set up and run the
documented training entrypoint, links from AGENTS.md resolve, fast checks pass,
and immutable inputs plus output provenance are enforced.

## Gotchas

- `data/raw/` is ground truth. Transformations write elsewhere.
- Catch only expected data or environment failures; fail early with context for
  everything else.
- Readability beats abstraction in experiments; maintenance boundaries beat
  copy-paste in long-lived training projects.
