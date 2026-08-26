# Machine-Learning Experiment Records

Read only when the project trains, evaluates, compares, promotes, or
stores machine-learning models.

## GitHub is the index and the gate, never the tracker or the store

GitHub has **no experiment tracking and no model registry** (GitHub Models
was a retired third-party inference catalog, not a registry — never cite
it). Preserve the project's existing tracker (MLflow, W&B, DVC, or
similar); do not migrate it, and do not simulate a tracker out of issues.
GitHub's role is what it is natively good at: the code and config live in
the repository, the run record is committed, promotion is a pull request
plus a release, and access is repository permissions.

This branch also **contradicts the Actions default deliberately**: hosted
runners have no GPUs, minutes are billed, and artifacts expire (90/400-day
caps) — training does not run on default Actions infrastructure, and an
Actions artifact is never an experiment store. CI's role in ML is the
cheap part: lint, unit tests, config validation, and metadata checks.

## The committed run record

Every run that matters gets a committed record
(`assets/experiment-record.md` is the starting
shape): source commit, dependency lock or image digest, dataset
identifiers with immutable versions or checksums, preprocessing version,
configuration and hyperparameters, seeds, hardware and runtime identity,
metrics with their definitions and evaluation-set identity, artifact
locations with immutable identifiers, and start/end/status. The tracker
holds the interactive view; the repository holds the durable identity.
Never commit weights, restricted datasets, or secrets to make them
discoverable — record governed locations instead.

## Storage and promotion

Choose storage deliberately: Releases (draft-first under immutable
releases, 2 GiB per file) for versioned model artifacts consumed by
humans; `ghcr.io` for OCI-packaged models consumed by deployment; Git LFS
within plan limits for in-repo assets; the existing external store for
everything larger. For whichever is chosen, settle retention, cleanup,
read/write access, and the owner of each store — preserving an existing
tracker does not defer these.

Promotion is a reviewed pull request that updates the promoted-model
reference plus, where releases are the vehicle, a release; the record
names the metric definition and threshold agreed **before** the run.
Rollback is re-promoting the previous record. Data and imagery
confidentiality rules from the design tree apply to issue text, PR text,
and CI logs alike.

Done when: a future agent can trace any promoted model to code,
environment, data, configuration, evaluation, and approval evidence using
only the repository and the named stores — and no rule pretends GitHub
tracks experiments.
