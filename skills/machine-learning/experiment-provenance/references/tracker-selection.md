# Tracker Selection and Wiring

Read when choosing or wiring a tracker, or when the project has none.

## Precedence

1. **A tracker the project already uses and that works** stays. Add the
   manifest fields to it; do not migrate.
2. **The hosting platform's experiment tracking**, when the project is
   hosted on a platform that provides it (GitLab's machine-learning
   experiments, reached through its MLflow-compatible client). Verify on
   the target instance that the feature is enabled and which client
   operations it supports before adopting it.
3. **A lightweight open tracker** otherwise — Trackio is the default for a
   project with no platform tracker: local-first, minimal dependencies,
   and shareable without a server.

Verify the selected tracker's current API and its integration with the
training helper (an acceleration library's `log_with` option or an
explicit client) from first-party documentation at wiring time; do not
recall method names.

## What the tracker receives

| Content | As |
|---|---|
| The manifest's scalar fields (run id, commit, dirty flag, image or lock digest, seeds, host facts) | parameters or tags, so runs can be filtered by them |
| The resolved configuration | a parameter set or an attached file; both when the tracker supports it |
| Metrics | logged against the global step, never against wall-clock only |
| Artifact locations | as strings with immutable identifiers; the artifacts themselves go to the governed store unless the tracker is that store |
| The manifest file | attached when the tracker stores files |

Wire it through one seam in the training loop — the single function that
logs metrics — so that switching trackers changes one place. The manifest
is written at start with status running and finalized at the end.

## What the tracker never receives

Credentials, presigned URLs, raw samples, prompts, full environments
(`os.environ`), or tensor contents. Identifiers and hashes stand in for
them. A tracker is a shared surface; treat it as published.

## Retention and access

Whatever tracker is kept or adopted, settle who owns it, how long runs
and artifacts are retained, and who may read and write, and record those
answers in the project's conventions. Adopting a tracker does not defer
these.
