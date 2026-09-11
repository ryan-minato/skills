# Provenance and the Tracker

Read when wiring the manifest and the tracker, or when the project
already uses a tracker.

## The run equation

```text
run = source snapshot + resolved configuration + environment identity + input identities
```

plus a run id minted at start. Every part is immutable once recorded;
a branch, `latest`, a path, or a Dockerfile is a name, not an identity.

## The manifest

`run_manifest.py` is copied unchanged into
the project (it mirrors the durable provenance skill's module).
`train.py` calls `start_manifest` before the first step and
`finish_manifest` at the end; the start-time record is kept as
`manifest.running.json`. It records the executed commit and dirty flag
(with a hash of the uncommitted changes, untracked files included), the
resolved configuration's hash, the image digest from `IMAGE_DIGEST` or
the runtime lock's hash, interpreter, host, GPU, and runtime facts, the
input identities the entry point passes, the seed, and the parent run.
Missing identities land in `degraded` and on stderr; a degraded run is
cited as degraded, never as complete.

## Snapshot and retention

- Commit before every run so the executed source is the recorded source;
  a dirty tree launches nothing (the manifest would mark it degraded).
- Research runs happen on an experiment branch or worktree, never on the
  integration branch.
- Cited snapshots stay reachable after a squash or a branch deletion:
  the project chooses a tag per run (`run/<run_id>`) or kept research
  branches, and `AGENTS.md` records the choice.

## Tracker precedence

1. A working tracker the project already uses stays; add the manifest
   fields to it.
2. The hosting platform's experiment tracking when the platform provides
   one — GitLab's machine-learning experiments through its
   MLflow-compatible client; verify on the instance that the feature is
   enabled.
3. Trackio otherwise: local-first, minimal dependencies, shareable
   without a server.

Wire it through the loop's single logging seam (`log_metrics`); pass the
tracker to `Accelerator(log_with=...)` and call `accelerator.log` there.
The manifest's scalar fields go to the tracker as parameters and tags;
metrics are logged against the global step. Verify the tracker's current
API and its Accelerate integration from first-party documentation at
wiring time. Never in the tracker: credentials, presigned URLs, raw
samples, prompts, tensor contents. Record in `AGENTS.md` which tracker
holds runs and where its view lives.
