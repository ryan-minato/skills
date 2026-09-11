---
name: experiment-provenance
description: >-
  Records and judges the identity of one machine-learning run — the
  executed source snapshot, the resolved configuration, the environment
  identity, the input identities, and a run id distinct from the commit —
  and wires the tracker that holds them. Use when the question is what a
  run must record or whether it can be reproduced: "results differ between
  runs and I can't tell what changed", "what should every run record",
  "which run produced this checkpoint"; when adding, choosing, or wiring an
  experiment tracker; when a run record names a branch, `latest`, or a
  Dockerfile as an identity; or when someone wants to edit a past run's
  record. Not for planning or organizing a series of experiments,
  hypotheses, or the pull request that carries them (the research-task
  role), for scaffolding a new project, or for build provenance of
  software artifacts outside machine learning.
license: Apache-2.0
---

# Experiment Provenance

Precedence on every run: an explicit user instruction, then the project's
own conventions and constraints, then this skill's defaults, then tool
preferences. A working tracker, configuration system, or record format is
preserved; a default here is never permission to migrate.

## The run equation

A run that supports a comparison, a decision, or a research claim must be
able to answer what code ran, with what configuration, on what inputs, in
what environment, and what came out. Its identity is a composition:

```text
run = source snapshot + resolved configuration + environment identity + input identities
```

plus a **run id** minted when the run starts. Every part is immutable once
recorded. A part that can change meaning later — a branch, a tag such as
`latest`, a path, a Dockerfile — is not an identity; resolve it to the
immutable form before recording.

## Run identity

- Mint the run id at start (a random or time-ordered unique id). Never
  reuse one; a rerun is a new run with its own id.
- The run id is not the commit. One source snapshot produces any number of
  runs; the commit identifies code, the run id identifies an execution.
  Human-readable names (`lr1e-4_seed42`) are labels beside the id, never
  the id.
- Run history is append-only. A wrong record is corrected by a new record
  or a correction note that points at the original, never by rewriting it.
  Refuse a request to "fix" a completed run's configuration in place.

## Source snapshot

- The source identity is the repository plus the commit that actually
  executed. Record the commit and whether the tree was dirty; a dirty tree
  means the recorded commit is not the executed source.
- Before launching from a tree with uncommitted changes, make a snapshot
  commit on the experiment branch or worktree so that recorded equals
  executed. When a snapshot cannot be made, record the patch hash of the
  uncommitted diff and mark the run as degraded.
- Referenced snapshots stay reachable. When the branch will be squashed or
  deleted, the project needs a retention rule — a tag per run
  (`run/<run_id>`) or a kept ref — so garbage collection never drops a
  commit a run record cites. Record the rule in the project's conventions.
- Git history describes code evolution, not runs: one commit per run is
  not required, and a run is never represented as a commit.

## Resolved run configuration

Save the configuration in effect after every source has been merged and
every derived value resolved:

```text
defaults + project config + experiment config + CLI overrides
+ search suggestions + runtime-derived values
→ resolved run configuration
```

- The command line, a partial override, or the raw YAML is not the record;
  the merged document is. Save it beside the manifest in the run's output
  directory and log it to the tracker as a parameter set or an attached
  file; the manifest carries its path and hash.
- Once training starts the resolved configuration is immutable input.
  State that changes during training under the configuration's own rules —
  a scheduled learning rate, a curriculum stage — is training state, not a
  configuration change.
- Vocabulary: "hyperparameter" means a value that is one in the usual sense
  (learning rate, weight decay, dropout, optimizer betas, warm-up). A
  dataset choice, an optimizer family, an architecture variant, a seed, or
  a device count is data, algorithm, model, randomness, or resource
  configuration; when such values are searched, call them **search
  variables**. Group the document by model, data, training, evaluation,
  randomness, runtime, resource, and operational configuration.

## Inputs and lineage

- Every input that affects the result has an identity: dataset,
  pretrained model, previous checkpoint, tokenizer, benchmark definition,
  external artifacts. The identity is enough to determine what was used
  and to fetch or rebuild it: a dataset revision or content checksum, a
  model repository plus commit, an object version id plus checksum.
- Mutable references are resolved before the run and the resolved value
  is what gets recorded: `dataset:main` → a revision, `model:latest` → a
  commit, `image:latest` → a digest.
- A run that starts from a checkpoint or a produced model records the
  parent run's id, so stages form a lineage instead of unrelated runs.

## Environment identity

- A container image is the strongly recommended environment carrier;
  its identity is the **image digest**, never the Dockerfile (a recipe)
  or a tag (mutable). Read the digest from the pushed image's repository
  digests, or from the build's image-id file when the image is never
  pushed, and inject it into the run (`IMAGE_DIGEST`) so the manifest can
  record it.
- Without a container, the environment identity is the dependency lock
  digest plus the interpreter version. A requirements file with ranges
  is not a lock: it identifies nothing, and the record says so.
- A container does not pin the host. For a quality claim the image digest
  and lock digest suffice; for a performance claim also record the GPU
  model and count, the driver, the runtime versions (CUDA or ROCm, the
  collective-communication library), and the host or node.

## The manifest

Write the manifest when the run starts (status running) and finalize it
when the run ends (status and end time); keep both versions (the module
keeps the start-time record as `manifest.running.json`). A manifest that
lacks an identity — a dirty tree, no resolved configuration, no image or
lock digest — carries the reason in `degraded` and is cited as degraded,
never as complete. Emit it to
the tracker as parameters and tags and to the run's output directory as a
file. [`assets/run_manifest.py`](assets/run_manifest.py) is a drop-in
standard-library module that collects the fields below and writes them;
copy it into the project and call it from the training entry point.

| Field | Source |
|---|---|
| `run_id`, `started_at`, `ended_at`, `status`, `degraded[]` | the run |
| `source.commit`, `source.dirty`, `source.patch_sha256` (dirty only: tracked diff plus untracked files), `source.remote` | git |
| `config.resolved_path`, `config.sha256` | the resolved configuration dump |
| `environment.image_digest` or `environment.lock_sha256` (a real lock file, or a fully pinned requirements file), `environment.python` | image, lock file, interpreter |
| `host.hostname`, `host.platform`, `host.gpus` (model, count), `host.driver`, `host.runtime` | the machine |
| `inputs[]` — `name`, `kind`, `identity` | resolved before the run |
| `randomness.seed`, `randomness.deterministic` | the run |
| `parent_run_id` | when resumed or fine-tuned from a run |
| `tracker.run_url` | the tracker |

Read [references/run-record.md](references/run-record.md) when a run is
cited as evidence in a pull or merge request, a report, or a promotion, or
when a committed run record is required; the committed record's shape is
[`assets/run-record.md`](assets/run-record.md).

## Where the facts live

- The tracker holds runs: metrics, the manifest as parameters and tags,
  artifact locations. Read
  [references/tracker-selection.md](references/tracker-selection.md) when
  choosing or wiring a tracker, or when the project has none.
- The repository holds code and configuration; the pull or merge request
  holds the decision and links to the evidence; the project's agent
  guidance holds the conventions (the retention rule, the manifest
  location, the tracker name). Each fact has one home; the others point.
- Never in a tracker, a manifest, or a log: credentials, presigned URLs,
  raw samples, prompts, or tensor contents. Record identifiers and hashes;
  raw material goes to a governed artifact store with access control.

## Handoffs

- Organizing several hypotheses, runs, and a pull or merge request into
  one research task is the research-task role. This skill pairs with
  `research-workflow` for it. If it is not installed, load the
  `ryan-minato-skills-installing` skill and install `research-workflow`
  as it directs; never run an install command yourself. (If that installer
  skill is absent too, it lives in the `core` catalog of
  https://github.com/ryan-minato/skills.) If the user declines, answer the
  run-record question alone and say that task organization was left out.

## Gotchas

- `main`, `latest`, `HEAD`, and a working-directory path are names, not
  identities; the validator of a record is "could this resolve to
  something else next week?".
- A seed does not make a run deterministic: record the framework's
  determinism flags and the nondeterministic operations the project
  accepts, or a "same seed" comparison proves nothing.
- A presigned URL is both a credential and a mutable reference; record the
  object's version id and checksum instead.
- A tracker's own "latest" or "best" alias moves; cite the run id.
- Recording the Dockerfile's hash records the recipe, not the environment:
  the same file builds different images on different days.
