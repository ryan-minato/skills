## Context

See proposal.md for motivation. Five new public skills in a new catalog
`skills/machine-learning/`, whose `CONTEXT.md` (companion change) grants
dependencies on `core` only and no grant between siblings: every sibling
pairing is an optional handoff named by role, routed through
`ryan-minato-skills-installing` with the template in
`.agents/knowledge/skill-quality.md`, never an install command. The catalog
is durable and installed per project, so no disposable marker and no name
prefix; names follow the default `<subject>-<action>` shape. Binding
limits: `description` ≤ 1024 characters (warn above 900), body under 500
lines, references split by branching condition with a precise load
sentence, no path outside the skill directory, no documentation-URL
index, volatile tool facts verified from first-party sources at use time,
English only. Script rules for `scan_series.py`: `--help`, exit codes
0/1/2 with deviations documented, data on stdout, diagnostics on stderr,
idempotent, standard library only. The source material is three Chinese
documents the maintainer supplied in conversation (an ML experiment
engineering standard, a training-telemetry catalog with severity and
root-cause procedures, a training-health metrics catalog); they are
unpublished, so the skills carry their substance in English, never a
translation, and cite no source. Nothing in
the repository mentions trackers, resolved configuration, research specs,
or training telemetry today, so no existing text is superseded.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| experiment-provenance — Trigger: description | `SKILL.md` frontmatter `description` | — |
| experiment-provenance — Behavior: four immutable parts plus a run id | `SKILL.md` `## The run equation`, `## Run identity`, `## Environment identity`; `references/run-record.md` (canonical field list); `assets/run-record.md`, `assets/run_manifest.py` | "Read `references/run-record.md` when a run is cited as evidence in a pull or merge request, a report, or a promotion, or when a committed run record is required." |
| experiment-provenance — Behavior: resolved configuration | `SKILL.md` `## Resolved run configuration` (sources merged, immutable after start, hyperparameter versus search variable, configuration groups, no rewriting of history) | — |
| experiment-provenance — Behavior: source snapshot and reachability | `SKILL.md` `## Source snapshot` | — |
| experiment-provenance — Behavior: tracker by precedence | `SKILL.md` `## Where the facts live`; `references/tracker-selection.md` | "Read `references/tracker-selection.md` when choosing or wiring a tracker, or when the project has none." |
| experiment-provenance — Handoff: research task organization | `SKILL.md` `## Handoffs` | — |
| research-workflow — Trigger: description | `SKILL.md` frontmatter `description` | — |
| research-workflow — Behavior: research spec | `SKILL.md` `## The research spec`; `references/research-spec-fields.md`; `assets/research-spec.md` | "Read `references/research-spec-fields.md` when writing or revising a research spec." |
| research-workflow — Behavior: one task, one request | `SKILL.md` `## The research task`; `references/branching-and-pr.md` | "Read `references/branching-and-pr.md` when opening the research worktree or branch, making snapshot commits, or preparing the pull or merge request that closes the task." |
| research-workflow — Behavior: hypothesis loop | `SKILL.md` `## The loop`; `assets/hypothesis-log.md`; `references/branching-and-pr.md` | as above |
| research-workflow — Behavior: evidence and completion | `SKILL.md` `## Evidence must match the claim`, `## Closing the task`; `references/evidence-and-search.md` | "Read `references/evidence-and-search.md` when deciding whether results support a claim, when comparing against a baseline, or when automatic search is on the table." |
| research-workflow — Behavior: automatic search | `SKILL.md` `## Automatic search`; `references/evidence-and-search.md` | as above |
| research-workflow — Handoff: run identity and recording | `SKILL.md` `## Handoffs` | — |
| experiment-code-conventions — Trigger: description | `SKILL.md` frontmatter `description` | — |
| experiment-code-conventions — Behavior: abstraction | `SKILL.md` `## Abstraction` | — |
| experiment-code-conventions — Behavior: loop and dependencies | `SKILL.md` `## The training loop`, `## Dependencies`; `references/vendoring-research-code.md` | "Read `references/vendoring-research-code.md` when copying code from a paper's or another project's repository into this one." |
| experiment-code-conventions — Behavior: configuration surface | `SKILL.md` `## The configuration surface`; `references/config-surface.md` | "Read `references/config-surface.md` when creating or restructuring the configuration system, or when a configuration file starts naming classes, registries, or conditionals." |
| experiment-code-conventions — Behavior: tests | `SKILL.md` `## Tests and hooks`; `references/tensor-tests-and-docs.md` | "Read `references/tensor-tests-and-docs.md` when writing tests or docstrings for tensor code, or when a type checker is proposed for it." |
| experiment-code-conventions — Behavior: style and performance | `SKILL.md` `## Style`, `## Performance and readability`; `references/tensor-tests-and-docs.md`, `references/hot-path-performance.md` | "Read `references/hot-path-performance.md` when readability and performance conflict in the step, data, or kernel path." |
| experiment-code-conventions — Handoff: training instrumentation | `SKILL.md` `## Handoffs` | — |
| training-instrumentation — Trigger: description | `SKILL.md` frontmatter `description` | — |
| training-instrumentation — Behavior: three layers, keys, instruments | `SKILL.md` `## Principles`, `## Correlation keys`; `references/framework-hooks.md` | "Read `references/framework-hooks.md` when wiring a specific framework's hooks: the PyTorch profiler, memory snapshot, or component logging, the TensorFlow profiler or debugger, or the JAX profiler, transfer guard, or NaN debugging." |
| training-instrumentation — Behavior: minimal set, labels, raw data | `SKILL.md` `## The minimal set`, `## Signal catalog`, `## Privacy`; `references/cluster-telemetry.md` | "Read `references/cluster-telemetry.md` when training spans more than one node, or when designing collection, storage, sampling, retention, or capacity for training telemetry." |
| training-instrumentation — Behavior: alerts | `SKILL.md` `## Severity and alerts`; `references/alerting-and-dashboards.md` | "Read `references/alerting-and-dashboards.md` when the request includes alerts, severity, on-call routing, or dashboards." |
| training-instrumentation — Behavior: model-health metrics | `SKILL.md` `## Signal catalog` (health rows), `## PyTorch skeleton`; `references/model-health-metrics.md`, `references/model-family-signals.md`; `assets/torch_health.py` | "Read `references/model-health-metrics.md` when instrumenting optimization dynamics or network internals beyond loss and gradient norm, or when asked what a specific health metric means." / "Read `references/model-family-signals.md` when the model is a transformer language model, uses mixture-of-experts, pipeline, or tensor parallelism, or trains with reinforcement-learning, adversarial, or diffusion objectives." |
| training-instrumentation — Handoff: training diagnosis | `SKILL.md` `## Handoffs` | — |
| training-diagnosis — Trigger: description | `SKILL.md` frontmatter `description` | — |
| training-diagnosis — Behavior: first anomaly, top-down | `SKILL.md` `## Method`; `scripts/scan_series.py` | — |
| training-diagnosis — Behavior: playbooks | `SKILL.md` `## Symptom index`; `references/throughput-and-utilization.md`, `references/memory.md`, `references/distributed-and-hardware.md`, `references/checkpoint-and-storage.md`, `references/generalization-and-data.md` | one sentence per reference in the symptom index, each naming the symptom family that loads it ("Read `references/memory.md` when a run hits an out-of-memory error, memory grows over time, or reserved memory far exceeds allocated.") |
| training-diagnosis — Behavior: replay | `SKILL.md` `## Numerical instability and replay`; `references/numeric-instability.md`; `references/deep-diagnosis-tools.md` | "Read `references/numeric-instability.md` when loss spiked, went NaN or Inf, diverged, or the precision scaler keeps skipping steps." / "Read `references/deep-diagnosis-tools.md` when the resident signals cannot separate the remaining causes and a profiler, memory snapshot, communication debug window, or replay must be switched on." |
| training-diagnosis — Behavior: evidence chain | `SKILL.md` `## The evidence chain`; `assets/evidence-chain.md` | — |
| training-diagnosis — Handoff: training instrumentation | `SKILL.md` `## Handoffs` | — |
| training-diagnosis — Script: scan_series.py | `scripts/scan_series.py`; `SKILL.md` `## Method` introduces it with a relative link at first mention | — |

## Description

Each description states its capability in the third person and its
triggers as "Use when …", with a "Not for …" clause that names the
neighbouring request the near-miss scenarios exercise. Budget: under 900
characters each. What each must contain:

- experiment-provenance: reproducibility of a run, what a run records,
  which run produced an artifact, wiring or choosing a tracker; direct
  phrasings ("make this run reproducible", "add experiment tracking") and
  indirect ones ("results differ between runs and I can't tell why"). Not
  for organizing a series of experiments, scaffolding a project, or
  software build provenance.
- research-workflow: organizing experiments or hypotheses toward an
  objective, writing or revising a research spec, judging whether results
  support a claim, closing a line of research; indirect phrasings ("let's
  plan the experiments", "how do we wrap this up"). Not for a software
  specification before a feature, team process design, or one run's
  record.
- experiment-code-conventions: the shape of training or experiment code —
  share or duplicate, Trainer or loop, vendoring, configuration contents,
  tests, typing, docstrings, comments, readability versus performance in a
  training path. Not for application code with no training or tensor
  concern or a refactoring request naming no ML code.
- training-instrumentation: what a training loop or job should log or
  measure, how often, correlating metrics, traces, and profiles, whether a
  metric's value is normal, alerts, severities, dashboards. Not for a run
  that already failed or slowed, or logging in a service that trains
  nothing.
- training-diagnosis: a run that failed, diverged, slowed, stalled, ran out
  of memory, or behaves unevenly across devices, asking why or how to fix
  it; symptom vocabulary (utilization, NaN, straggler, OOM, checkpoint
  time). Not for deciding what to log before an incident, or an
  environment or installation problem unrelated to a running job.

## Dependencies and handoffs

- `ryan-minato-skills-installing` (`core`, in range): the route for every
  handoff below, with the template from `skill-quality.md`.
- `sensitivity-check` (`core`, in range): named by `training-diagnosis`
  before a log excerpt leaves the machine; when absent, the agent reviews
  the excerpt itself and says so.
- Sibling handoffs, all by role in descriptions and specs (the body may
  name the sibling as the example of the role): experiment-provenance →
  the research-task role; research-workflow → the run-identity role;
  experiment-code-conventions → the instrumentation role;
  training-instrumentation → the diagnosis role; training-diagnosis → the
  instrumentation role. Each fallback when the user declines is stated in
  its `Handoff:` requirement.
- No dependency on `scaffold`, `meta`, `engineering`, or any other
  repository. Spec tooling for research tasks is left to the harness
  builders; `research-workflow` ships a carrier-neutral skeleton only.

## External impact

- Catalog files, root documents, labels, issue forms, marketplace entry,
  symlinks, and the entropy review: the companion repository change
  `machine-learning-catalog-harness` (its design names the proof per
  item). Proof here: `just validate` green with the five skill directories
  present.
- README pair rows for the five skills in `skills/machine-learning/README.md`
  and `README.zh.md`: content-identical pairs; proof by reading both.
- `.claude-plugin/marketplace.json` `machine-learning` plugin `skills[]`:
  `just gen-marketplace` after each skill lands; proof `just validate`.
- No existing skill, knowledge file, project skill, or mirror in
  `scripts/validate_harness.py` changes; proof `git diff --stat
  origin/main...HEAD -- skills/core skills/engineering skills/meta
  skills/scaffold skills/writing scripts .agents` is empty apart from the
  five symlinks.

## Decisions

- **Five skills split by activity, not by source document** (serves every
  Trigger requirement): design-time instrumentation versus incident-time
  diagnosis, one run's identity versus a research task's process, and code
  shape as its own unit give each description a distinct trigger surface;
  the health-metrics document is split by column (definition, normal
  range, frequency, alert strategy → instrumentation; abnormal pattern,
  ranked causes, fix, replay → diagnosis) so thresholds live in one place.
  Alternative rejected: one skill per document, whose triggers would
  overlap on NaN, gradient norm, and GPU utilization vocabulary.
- **Cluster-side telemetry stays a reference of `training-instrumentation`**
  (serves Behavior: minimal set): it is the same design-time activity for
  a different scale; a sixth skill is the fallback only if the body cannot
  stay under budget after drafting.
- **One bundled script, `scan_series.py`** (serves Behavior: first anomaly
  and Script: scan_series.py): locating the first sustained anomaly is
  deterministic arithmetic an agent otherwise re-derives by hand; the
  alert rules, the capacity model, and the manifest writer earn no
  four-scenario contract (rules are backend configuration, the model is
  one line, the manifest writer is a copied asset). Exit 1 for unusable
  input or a missing column is documented in `--help`.
- **Assets are drop-in modules and skeletons, not templates with
  placeholders**: `run_manifest.py` and `torch_health.py` run as written
  in a project; `research-spec.md`, `hypothesis-log.md`, `run-record.md`,
  `evidence-chain.md` are section skeletons the agent fills. Alternative
  rejected: `{{PLACEHOLDER}}` templates, which belong to the disposable
  builders' contract, not to durable skills.
- **Tracker precedence (keep existing → platform's tracking on GitLab →
  Trackio)** (serves Behavior: tracker): follows the source standard;
  every tracker API detail is verified at use time and the reference
  names capabilities, not commands.
- **The canonical run-record field list lives in
  `experiment-provenance/references/run-record.md`**: the follow-up change
  aligns the platform builders' record assets to it and registers the
  mirror; this change creates the source of truth only.
- **Every scenario keeps its Trigger prompt verbatim as the test prompt**:
  the delta specs were written as executable prompts so the verification
  plan copies them rather than paraphrasing.

## Risks / Trade-offs

- [`training-instrumentation` body exceeds the 500-line warning] → catalog
  as tables, every conditional branch in a reference; if `just check-skill`
  still warns, split the cluster reference into a sixth skill in a
  follow-up, not here.
- [Descriptions overlap on shared vocabulary (NaN, gradient norm, spec,
  workflow)] → each description carries a "Not for …" naming the
  neighbour; the near-miss Trigger cases test both directions between
  instrumentation and diagnosis.
- [Trigger solvers inside this repository answer from `.agents/knowledge/`
  and see every repository skill] → solvers run in a fixture project
  outside the repository with the candidate skills copied to the user
  skills directory for the run and removed afterwards; only the target's
  own load decision is graded.
- [Floor-tier solvers load skills inconsistently] → Sonnet-class is the
  floor these skills claim; certification covers that tier only.
- [Test budget is limited by the maintainer] → Trigger cases run one load
  prompt and one near-miss per skill; the remaining Trigger, Behavior, and
  Handoff scenarios are verified by a clean-context readback of the skill
  text against each scenario and recorded as not executed, with the
  reason, in the Validation section.
- [Tool facts drift (Trackio, GitLab experiments, profiler APIs, OpenTelemetry)]
  → references name capabilities and the first-party source to verify at
  use time; no URL index.

## Verification plan

Solver tier: Sonnet-class (the least capable tier the skills claim).
Observation: framework-native skill-load history when available, else the
appended neutral `SKILLS_LOADED:` self-report. Isolation: fresh
clean-context subagent per prompt, a throwaway fixture project under the
session scratch directory (a minimal PyTorch training repository with a
`train.py`, a `configs/` directory, and a README), the five candidate
skills copied to the user skills directory for the run and removed
afterwards; one attempt per case, up to three on an invalid observation.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| experiment-provenance — Trigger: Reproducibility question | the scenario's prompt, verbatim, in the fixture | loads `experiment-provenance` (critical) | 1/1 | Sonnet-class | as above | as above |
| experiment-provenance — Trigger: Research task organization (near-miss) | the scenario's prompt, verbatim | does not load `experiment-provenance` (critical) | 1/1 | same | same | same |
| research-workflow — Trigger: Organizing a research task | the scenario's prompt, verbatim | loads `research-workflow` (critical) | 1/1 | same | same | same |
| research-workflow — Trigger: Software feature spec (near-miss) | the scenario's prompt, verbatim | does not load `research-workflow` (critical) | 1/1 | same | same | same |
| experiment-code-conventions — Trigger: Trainer or loop | the scenario's prompt, verbatim | loads `experiment-code-conventions` (critical) | 1/1 | same | same | same |
| experiment-code-conventions — Trigger: Web handler cleanup (near-miss) | the scenario's prompt, verbatim | does not load `experiment-code-conventions` (critical) | 1/1 | same | same | same |
| training-instrumentation — Trigger: What to log | the scenario's prompt, verbatim | loads `training-instrumentation` (critical) | 1/1 | same | same | same |
| training-instrumentation — Trigger: Incident already happening (near-miss) | the scenario's prompt, verbatim | does not load `training-instrumentation` (critical) | 1/1 | same | same | same |
| training-diagnosis — Trigger: Numerical failure | the scenario's prompt, verbatim | loads `training-diagnosis` (critical) | 1/1 | same | same | same |
| training-diagnosis — Trigger: Designing signals (near-miss) | the scenario's prompt, verbatim | does not load `training-diagnosis` (critical) | 1/1 | same | same | same |

Readback cases (clean-context subagent reads the finished skill directory
and, for each scenario below, quotes the passage that produces the
scenario's THEN and states whether it is present, precise, and
unconditional; a scenario with no passage is a critical failure; threshold:
every scenario has a passage):
- experiment-provenance: Tracker wiring; Software build provenance
  (near-miss); every Behavior scenario; Handoff offered; User declines.
- research-workflow: Closing with a negative result; Team process design
  (near-miss); every Behavior scenario; Handoff offered; User declines.
- experiment-code-conventions: Configuration doing too much; Generic
  refactoring (near-miss); every Behavior scenario; Handoff offered; User
  declines.
- training-instrumentation: Alerts for training jobs; Service logging
  (near-miss); every Behavior scenario; Handoff offered; User declines.
- training-diagnosis: Utilization drop; Import failure (near-miss); every
  Behavior scenario; Handoff offered; User declines.

Script and tool harnesses (`S=skills/machine-learning/training-diagnosis/scripts/scan_series.py`,
run from an untracked harness directory outside version control holding a
generated `series.csv` with a stable prefix and a sustained spike, plus a
grouped variant with a `rank` column):
- Help: `python3 $S --help` → usage on stdout naming `--input`, `--column`,
  `--window`, `--z`, `--k`, `--m`, `--group-by`, and the exit codes; exit
  0.
- Representative run: `python3 $S --input series.csv --column loss` → JSON
  on stdout with `first_anomaly_step`, its score, and the window
  parameters; exit 0. Grouped: `--group-by rank` adds the
  slowest-to-median ratio per step.
- Repeated run: the representative command twice into files; `diff`
  empty; the harness directory's listing unchanged.
- Bad arguments: `python3 $S --bogus` → stderr names `--bogus`; exit 2.
  Missing column: `--column nope` → stderr names the column; exit 1.
- `just check-skill skills/machine-learning/<name>` for each skill, `just
  lint`, `just spec-validate`, `just check`.

Skipped (recorded in the Validation section with this reason):
- Every Trigger scenario not in the table above, and every Behavior and
  Handoff scenario, is not executed by a solver; the maintainer limited
  the test fleet to one load and one near-miss prompt per skill. Each is
  covered by the readback case instead.

## Open Questions

None.
