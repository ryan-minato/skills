---
name: experiment-code-conventions
description: >-
  Shapes machine-learning experiment code — when to share or duplicate, an
  explicit training loop over a Trainer, vendoring research repositories,
  a configuration surface that stays values rather than a language, tests
  as behavior contracts with a light CPU default suite and GPU-only tests
  that fail without hardware, hooks that never run tests, near-default
  lint with no global type gate over tensor code, and hot-path performance
  kept over readability. Use when writing, reviewing, or restructuring
  training or experiment code: "should these two scripts share a Trainer",
  "the config names classes by import path", "add mypy to the model
  code?", "depend on this paper's repo or copy it?", "skip GPU tests when
  there is no GPU?". Not for application code with no training or tensor
  concern, or a refactoring request that names no machine-learning code.
license: Apache-2.0
---

# Experiment Code Conventions

Precedence on every change: an explicit user instruction, then the
project's own conventions and constraints, then this skill's defaults,
then tool preferences. Experiment code serves credible experiments, not
the code itself: optimize reproducibility, readability, and comparability
over the abstraction and generality a service codebase would want.

## Abstraction

Abstract **semantic coupling**, never accidental similarity:

- Share an implementation only when two places must stay logically
  consistent and will change together — the same preprocessing in the
  training and evaluation scripts is the canonical case, because a
  drifted copy silently corrupts the comparison.
- Tolerate duplication between code that merely looks alike today and
  serves different hypotheses; the copies are free to diverge with the
  research.
- The rule in one line: **abstract the stable mechanism, duplicate the
  unstable policy.** An abstraction says "these must stay the same"; it
  never says "these happened to be the same today".
- When the user cites repetition alone, ask whether the copies must stay
  consistent, explain this once, and follow their reaffirmed decision.
- Avoid indirection layers built to remove repetition: cross-file hooks,
  registries, and base classes that make the experiment unreadable top
  to bottom cost more than the lines they save.

## Dependencies

- Prefer libraries maintained by an organization with a track record,
  active maintenance, wide use, and a test suite (framework first-party
  packages, the major model-hub libraries, the mature core of a
  framework's ecosystem).
- An individual's or a lab's research repository, or code nobody has
  maintained for a long time, is not a runtime dependency: vendor the
  part that is needed. Read
  [references/vendoring-research-code.md](references/vendoring-research-code.md)
  when copying code from a paper's or another project's repository into
  this one.

## The training loop

The loop is research semantics, so keep it explicit. Gradient
accumulation, loss normalization, when gradients are zeroed, when the
scheduler steps, the mixed-precision boundaries, and when evaluation and
checkpointing happen all change results; a loop that shows them in order
beats a Trainer that scatters them across callbacks and hooks.

- Default: an explicit loop with an acceleration library handling
  devices, distribution, precision, and accumulation (Accelerate for
  PyTorch projects).
- A structured Trainer is acceptable when the training semantics are
  settled, are not a research variable, and the team knows its behavior.
- Keep one logging seam (the single function that emits metrics) and one
  stage-trace seam (data, forward, loss, backward, optimizer) in the loop,
  so instrumentation changes one place.

## The configuration surface

Configuration is the set of choices the project intends to expose to a
run: hyperparameters, dataset, architecture, algorithm, evaluation,
runtime, and resource choices. Its job is to let a run pick a state, not
to describe how the program is built. Read
[references/config-surface.md](references/config-surface.md) when
creating or restructuring the configuration system, or when a
configuration file starts naming classes, registries, or conditionals.

- Default for an unsettled project: a typed schema (dataclasses) plus
  YAML files plus command-line overrides, merged into one resolved
  configuration that is dumped with the run (OmegaConf).
- A configuration may select among named, code-supported choices
  (`optimizer: adamw | sgd`). It may not construct arbitrary objects,
  compose through registries or import paths, carry control flow, or
  inherit deeply — that is a second programming language.
- Something being "model configuration" does not make it runtime
  configuration: expose a value only when the project intends it to vary
  or the environment forces it to.
- A project that already uses a composition framework keeps it, with two
  rules: no object instantiation from configuration, and a pinned output
  directory.

## Tests and hooks

Tests are behavior contracts for the modules the project is responsible
for, not a coverage target.

- Test the project's integration assumptions — tensor shapes, mask
  semantics, padding, label alignment, reduction semantics, custom
  layers and adaptation code — not the third-party library underneath.
- The default suite is light and CPU-compatible and runs in ordinary CI.
- GPU-only tests are legitimate for behavior that exists only on the GPU
  path. They form a separate suite that runs only by an explicit command
  and **fails** when the hardware is absent; converting them to skips or
  CPU fallbacks hides broken tests.
- Expensive validation (full training, large data, equivalence against a
  reference at scale) runs by hand, never in the default suite or hooks.
- Git hooks run the formatter and the linter at most. Commits are
  experiment snapshots, and a hook that runs tests taxes every snapshot.
  Read [references/tensor-tests-and-docs.md](references/tensor-tests-and-docs.md)
  when writing tests or docstrings for tensor code, or when a type
  checker is proposed for it.

## Style

- Linter and formatter near their defaults (Ruff, line length 120,
  docstring code formatting on); when a rule keeps fighting the project's
  tensor idioms, disable the rule instead of scattering suppressions.
- No global static type gate over tensor-heavy code: the contracts that
  matter there — shape, dtype, device, layout, semantic dimensions, mask
  conventions — are not what a Python type checker checks, and forcing it
  clean spends effort on casts and protocols without making an
  experiment more credible. Static checking may guard configuration and
  control-plane modules that benefit from it.
- Write annotations that aid reading; document shape, dtype, device,
  layout, and mask conventions in docstrings or a nearby comment.
- Self-documenting code first; block-level comments for mathematical
  intent, non-obvious invariants, numerical reasons, layout tricks,
  distributed behavior, and performance rationale — never a line-by-line
  paraphrase.

## Performance and readability

Prefer readability wherever it costs nothing (names, file layout, control
flow). In the training and inference hot paths, do not accept a
meaningful performance loss for form: recover understandability with
local encapsulation, a block comment, a behavior test, a benchmark, and a
clear slower reference implementation that doubles as documentation and
numerical oracle. Read
[references/hot-path-performance.md](references/hot-path-performance.md)
when readability and performance conflict in the step, data, or kernel
path.

## Handoffs

- What the loop should emit and how its metrics are judged is the
  instrumentation role. This skill pairs with `training-instrumentation`
  for it. If it is not installed, load the `ryan-minato-skills-installing`
  skill and install `training-instrumentation` as it directs; never run
  an install command yourself. (If that installer skill is absent too, it
  lives in the `core` catalog of https://github.com/ryan-minato/skills.)
  If the user declines, keep the single logging seam and the single
  stage-trace seam in the loop and leave the metric design to the user.

## Gotchas

- A Trainer hides the step: the order of zeroing, accumulation, clipping,
  stepping, and scheduling is where results silently change.
- A configuration key that names an import path (`_target_`, `class:`)
  has turned configuration into code; replace it with a named choice.
- `skipif(no GPU)` on a GPU-only test makes the suite green on every
  machine that cannot run it; exclusion from the default suite plus
  failure under the explicit suite is the honest shape.
- Mocking tensor operations tests the mock. Use small real tensors on
  the CPU.
- Deduplicating two research scripts into a base class couples two
  hypotheses; the next change to one breaks the other.
- Coverage numbers say nothing about whether a shape contract holds.
