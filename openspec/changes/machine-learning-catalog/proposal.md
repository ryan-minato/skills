## Why

The library has no durable guidance for the daily work of a machine-learning
experiment project: recording what a run actually used, organizing a series
of hypotheses into one reviewable research task, shaping training code and
its configuration surface, deciding what a training run emits and when to
alert, and diagnosing a run that misbehaves. `scaffold-ml` initializes such
a project and is disposed of; the platform builders record only the
committed run record. Three source documents — an engineering standard for
ML experiment projects, a catalog of training telemetry with severity and
root-cause procedures, and a catalog of training-health metrics with their
diagnosis — now exist and are ready to become skills. Now, because the next
ML project built from this library will otherwise receive a scaffold with
no durable skills behind it, and because the scaffold and the `meta`
builders will be aligned to the same standard in a follow-up change that
hands off to these skills by role.

## What Changes

- New catalog `machine-learning`: durable, per-project skills for projects
  that train or evaluate models. Installed individually, never carrying the
  disposable marker; dependencies on `core` only; siblings named by role.
- `experiment-provenance`: an agent that loads it composes a run's
  identity from four immutable parts (source snapshot, resolved
  configuration, environment identity, input identities) plus a run id
  distinct from the commit, saves the resolved configuration and treats it
  as immutable after start, makes a snapshot commit before a run from a
  dirty tree and keeps referenced snapshots reachable, resolves mutable
  references to immutable ones, never rewrites run history, and selects a
  tracker (keep existing → GitLab experiments on GitLab → Trackio) that
  holds the manifest as params and tags and never secrets or raw samples.
- `research-workflow`: an agent that loads it writes a research spec with
  Objective and Evaluation always and Context, Search Scope, Constraints,
  Completion Condition, Hypotheses as needed; runs one research task per
  pull or merge request carrying many hypotheses, snapshot commits, and
  runs; keeps run history immutable while the spec evolves; requires
  evidence that matches the claim; recommends automatic search only when
  the space and the compute allow it; and closes a task on its completion
  condition, negative results included.
- `experiment-code-conventions`: an agent that loads it abstracts semantic
  coupling and tolerates accidental similarity, prefers mature first-party
  dependencies and vendors unstable research code with its origin, writes
  an explicit training loop (Accelerate by default), keeps the
  configuration surface to values a run may choose (OmegaConf typed schema
  plus YAML plus CLI overrides plus a resolved dump; existing Hydra kept
  without object instantiation), tests behavior contracts with a light CPU
  default suite and GPU-only tests that fail without hardware, keeps git
  hooks free of tests, applies near-default Ruff with no global type gate
  over tensor code, and keeps hot-path performance while recovering
  understandability elsewhere.
- `training-instrumentation`: an agent that loads it designs what a run
  emits across the outcome, optimization, and systems layers with shared
  correlation keys, separates metrics, stage-level trace, and scheduled
  profiling, selects a minimal signal set by priority with sampling
  frequencies, refuses unbounded metric labels and raw data in logs,
  designs symptom-first alerts relative to a healthy baseline with
  persistence and composite rules under five severities, and adds
  model-health metrics beyond loss, including model-family signals.
- `training-diagnosis`: an agent that loads it locates the first anomaly
  before reasoning about causes, walks a top-down ladder from correctness
  to performance, applies symptom-specific playbooks (utilization, memory,
  numerical instability, stragglers and communication, checkpoint and
  storage, hardware), confirms with a replay procedure, and reports an
  evidence chain; it ships `scripts/scan_series.py`, which finds the first
  sustained robust-z crossing in a metric series.

## Skills touched

- `machine-learning/experiment-provenance` (new): description triggers,
  run identity, resolved configuration, snapshots and reachability,
  tracker selection, the handoff to the research-task role.
- `machine-learning/research-workflow` (new): description triggers, the
  research spec, one task per pull or merge request, the hypothesis loop,
  evidence and completion, automatic search, the handoff to the
  run-identity role.
- `machine-learning/experiment-code-conventions` (new): description
  triggers, abstraction, dependencies and the loop, the configuration
  surface, tests and style, performance, the handoff to the
  instrumentation role.
- `machine-learning/training-instrumentation` (new): description
  triggers, layers and correlation, the minimal set and its limits, alert
  design, model-health metrics, the handoff to the diagnosis role.
- `machine-learning/training-diagnosis` (new): description triggers, the
  method, symptom playbooks, replay, the evidence chain, the handoff to
  the instrumentation role, the `scan_series.py` contract.

## Installed behavior

Every skill is new: an agent in a project that installs one gains a
capability it did not have → `feat` for each, scoped by skill name. No
existing installed skill changes in this change; the alignment of
`scaffold-ml` and the `meta` builders is the follow-up change
`ml-standard-alignment`.

## Impact

- New catalog scaffold `skills/machine-learning/README.md`, `README.zh.md`,
  `CONTEXT.md`; five README pair rows; five symlinks in `.agents/skills/`;
  a hand-added `machine-learning` plugin entry in
  `.claude-plugin/marketplace.json` followed by `just gen-marketplace`.
- Repository files that enumerate catalogs — `ARCHITECTURE.md`, the root
  `README.md` and `README.zh.md`, `.github/labels.json`, the three issue
  forms' Catalog options — and the entropy-review date in
  `.agents/knowledge/harness-maintenance.md`: the companion repository
  change `machine-learning-catalog-harness`.
- Mirrored files in `scripts/validate_harness.py`: none.
- No existing skill's handoffs, references, or assets change.

## Non-goals

- Reworking `scaffold-ml` to the same standard and aligning
  `meta-spec-workflow`, `meta-python-defaults`, `meta-gpu-container`,
  the platform builders' ML references, and `meta-workflow-design`: the
  follow-up change `ml-standard-alignment` on its own branch.
- A sixth, cluster-side telemetry skill: the platform material lives in a
  reference of `training-instrumentation` unless its body cannot stay in
  budget.
- Shipping an OpenSpec research-task schema from a durable skill: the
  paradigm builder configures spec tooling; `research-workflow` ships a
  carrier-neutral spec skeleton only.
- Any dependency grant between the five skills: they are installed one at
  a time and name one another by role.

## Tracked work

No issue: user-directed change planned in conversation.
