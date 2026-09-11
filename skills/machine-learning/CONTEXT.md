# machine-learning — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

Durable, per-project skills for the daily work of a project that trains or
evaluates models: recording what a run used, running a research task,
shaping experiment code, deciding what a training run emits, and
diagnosing a run that misbehaves. They are installed one at a time, stay
installed for the life of the project, and carry no disposable marker.

## Requirements

- Every skill opens its body with the same precedence: an explicit user
  instruction, then the project's own conventions and constraints, then the
  skill's defaults, then tool preferences. A working project choice is
  preserved; a skill's default is never permission to migrate.
- Guidance transfers across frameworks. PyTorch is the illustrative default;
  a rule must hold when the framework changes, and framework-specific hooks
  live in a reference behind a precise load condition.
- Thresholds are starting values. Any absolute number a skill gives is
  labelled as an initial value to recalibrate against the project's own
  healthy baseline; alert rules are expressed relative to that baseline.
- Volatile tool facts — tracker APIs, profiler options, platform features —
  are verified from first-party documentation at use time. No skill ships a
  documentation-URL index, a static tool inventory, or a version table.
- Nothing a skill records or logs may carry raw samples, prompts,
  credentials, presigned URLs, or tensor contents; identifiers and hashes
  stand in for them, and raw material goes to a governed store.
- Assets are drop-in modules or section skeletons that work as written;
  they carry no `{{PLACEHOLDER}}` slots, which belong to the disposable
  builders' contract.

## Dependencies

- Default range only: skills here may depend on `core` skills. No grant
  between the catalog's own skills and no grant to another catalog — they
  are installed one at a time, so co-presence is never guaranteed. A
  pairing between two skills here is an optional handoff named by role,
  routed through `ryan-minato-skills-installing`, with the fallback stated
  for when the user declines.
- No dependency on or recommendation of skills from other repositories; no
  exemptions.

## Naming

Default shape, no prefix or suffix: `<subject>-<action>`, where the
subject is the machine-learning object the skill acts on — `experiment`
(`experiment-provenance`, `experiment-code-conventions`), `research`
(`research-workflow`), `training` (`training-instrumentation`,
`training-diagnosis`).

## Scope

This catalog owns durable methodology for projects that train or evaluate
models, applied during the project's life. Initializing such a project —
its layout, environment, commands, checks, and agent guidance — belongs to
the disposable `scaffold` catalog's `scaffold-ml`; GPU container
environments to the `meta` catalog's `meta-gpu-container`; data pipelines
that consume models without training them to `scaffold-data-science`;
software specifications and the specify-plan-implement loop to
`engineering/spec-driven-development`; settling a project's spec tooling,
including a research-task schema, to the `meta` catalog's
`meta-spec-workflow`.

## Disambiguation

One run's identity — what it recorded, whether it can be reproduced,
which run produced an artifact, wiring a tracker → `experiment-provenance`
· a series of hypotheses toward one objective, the research spec, one task
per pull or merge request, evidence for a claim, closing with a verdict →
`research-workflow`; a software feature's specification → the
`engineering` catalog's `spec-driven-development`; how a team tracks and
plans work → the `meta` catalog's `meta-workflow-design` · the shape of
experiment code — share or duplicate, Trainer or loop, vendoring,
configuration contents, tests, typing, docstrings, hot-path trade-offs →
`experiment-code-conventions`; universal coding standards →
`programming-guidelines` in `core`; behavior-preserving restructuring of
any code → the `engineering` catalog's `code-refactoring` · what a run
emits, how often, correlation keys, alerts, severities, dashboards →
`training-instrumentation` · a run that failed, diverged, slowed, stalled,
ran out of memory, or is uneven across devices → `training-diagnosis`.

## References

_(none yet — add catalog-scoped reference URLs here)_
