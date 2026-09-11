# <project name>

<Two or three sentences: what gets trained and evaluated here, what the
model is for, and what counts as "better" (the benchmark).>

## Directory map

| Path | Holds | Rule |
|---|---|---|
| `train.py`, `eval.py` | workflow entries | run via just; `eval.py` is bound to the benchmark in `docs/data.md` |
| `config.py`, `configs/` | the configuration surface (schema + named states) | values and named choices only, never class paths |
| `<project_name>/` | shared code | created only when two places must stay consistent; absolute imports |
| `data/raw/` | immutable inputs (local cache) | never written by a transformation; never committed |
| `data/interim/`, `data/processed/` | derived data | regenerable; never committed |
| `outputs/<run_id>/` | resolved config, manifests, checkpoints, exports | never committed |
| `research/<task>/` | the research spec and hypothesis log | <or: the OpenSpec change directory under the `research-task` schema> |
| `docs/data.md` | sources, identities, the benchmark | updated with the work |

## Commands

| Command | Does |
|---|---|
| `just setup` / `just setup-train` | development / training-box environment from the committed lock |
| `just lock` | refresh the lock after editing dependencies (never edit a compiled file) |
| `just train [key=value ...]` | one run; overrides single configuration values |
| `just eval outputs/<run_id>` | evaluate a run on the recorded evaluation set |
| `just lint` | format + lint (Ruff) |
| `just test` | fast, CPU-compatible contract tests (the default suite) |
| `just test-gpu` | the accelerator suite — fails without hardware; run on `<training host>` |
| `just test-slow` | heavy validation — by hand only, never hooks or CI |

## Environment

- Dependency carrier: `<uv project (pyproject.toml + uv.lock) | requirements workflow (requirements*.in → *.txt)>`.
  The runtime lock (`<uv.lock | requirements.txt>`) is the environment
  identity a run records. Accelerator wheels: `<index routing in
  pyproject.toml | --torch-backend <backend> per machine: <machine> uses
  <backend>, <machine> uses <backend>>`. When hardware changes, revisit
  this first.
- <Container, if opted in: image `<name>`, built with `just docker-build`;
  the pushed digest (or `just docker-digest`) is exported as
  `IMAGE_DIGEST` before a run; volumes `data/` → `/app/data`, `outputs/`
  → `/app/outputs`, model cache → `/root/.cache/huggingface`; the image's
  framework is authoritative: <yes/no>.>

## Configuration

- Schema in `config.py`; named states in `configs/`; overrides on the
  command line (`optim.lr=1e-4`). Every run writes
  `outputs/<run_id>/config.resolved.yaml` before its first step — that
  file, not the command line, is the run's configuration.
- "Hyperparameter" means learning rate, weight decay, dropout, betas,
  warm-up; dataset, optimizer family, architecture, seed, and device
  count are search variables when searched.

## Provenance

- Every run writes `outputs/<run_id>/manifest.json` at start (kept as
  `manifest.running.json`) and finalizes it at the end: commit and dirty
  flag, resolved-config hash, `<image digest | lock hash>`, host and
  runtime facts, input identities, seed, parent run. A manifest with a
  `degraded` entry is cited as degraded, never as complete.
- Commit before every run; a dirty tree launches nothing. Research runs
  happen on an experiment branch or worktree. Cited snapshots stay
  reachable after a squash: `<tag run/<run_id> | keep research/<task>
  branches>`.
- Tracker: `<name>` (`<why: existing | platform | default>`), wired
  through `log_metrics` in `train.py`; runs are viewed at `<location>`.
  The tracker never receives credentials, presigned URLs, raw samples,
  or prompts.
- `latest`, a tag, a branch, a path, or a Dockerfile is a name, not an
  identity: record revisions, checksums, and digests.

## Research tasks

- A research task is one objective judged by one evaluation. Its spec
  (Objective and Evaluation required; Context, Search Scope, Constraints,
  Completion Condition, Hypotheses as needed) lives at
  `<research/<task>/spec.md | an OpenSpec change under the research-task
  schema (schema: research-task, skip_specs: true in .openspec.yaml)>`
  with the hypothesis log beside it.
- One task ↔ one pull or merge request carrying its hypotheses, snapshot
  commits, runs, and verdict; metrics stay in the tracker. The spec may
  evolve; run history is never rewritten, and a run belongs to the spec
  version it ran under. A task closes on its completion condition —
  negative results are valid outcomes.
- The research task is the project's unit of research work; hypotheses
  and runs are its contents, not work items.

## Code style

- Readability first, but never at a meaningful cost in the training or
  inference hot path; recover understandability there with local
  encapsulation, a block comment, a test, a benchmark, or a slower
  reference implementation.
- Share code only when two places must stay logically consistent;
  duplicate what may diverge with the research.
- Prefer mature, maintained first-party libraries; vendor unmaintained
  research code with its origin commit and license instead of depending
  on it.
- The training loop stays explicit (`train.py`); accumulation, clipping,
  scheduler stepping, precision, and checkpoint timing are visible there.
- Let it crash: catch only expected data or environment failures
  (counted and logged); everything else fails early with its traceback.
- Ruff formats and lints (line 120). No type checker over model code
  (shape, dtype, device, and mask contracts are not what it checks);
  static checking may guard `config.py` and control-plane modules.
  Docstrings on shared interfaces carry shape, dtype, device, and mask
  conventions; block-level comments explain mathematical intent, layout,
  and performance rationale.
- Profiling is switched on for a bounded window when a stage is
  anomalous, never left on.
- Tests protect behavior contracts (shapes, masks, padding, reductions,
  custom components), not coverage; `gpu` tests fail without hardware
  (a skip hides a broken test) and never enter the default suite;
  `slow` tests run by hand. Git hooks
  run the formatter and linter only — commits are experiment snapshots.

## Never

- Never write into `data/raw/`; never commit `data/`, `outputs/`,
  weights, checkpoints, or credentials (`.env` is ignored).
- Never wire `gpu` or `slow` tests into hooks or CI.
- <anything else agents must not do without asking>

## When to read what

| Situation | Read |
|---|---|
| touching data loading, sources, or the benchmark | `docs/data.md` |
| what a run must record; wiring or changing the tracker | the run-provenance skill (`experiment-provenance`), if installed; else the Provenance section above |
| planning a series of experiments; writing the research spec; closing a task | the research-task skill (`research-workflow`), if installed; else the Research tasks section above |
| shaping training code, configuration, tests, or a hot path | the experiment-code skill (`experiment-code-conventions`), if installed; else the Code style section above |
| deciding what the loop should log, alert on, or profile | the training-instrumentation skill, if installed |
| a run failed, diverged, slowed, or ran out of memory | the training-diagnosis skill, if installed |
