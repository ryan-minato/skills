# GitLab MLOps and Experiment Data

Read only when the project trains, evaluates, compares, stores, promotes, or
deploys machine-learning models.

Inspect the current experiment tracker, model registry, artifact/data stores,
training environment, pipeline, and provenance rules. Preserve a coherent
existing platform unless the user explicitly approves migration.

Resolve `machine learning model experiments`, `MLflow client compatibility`,
`model registry`, `package registry`, and `model deployment` through
`llms.txt`; confirm target-instance support and feature status.

## Experiment identity

Every run must record enough identity to reproduce and compare it:

- source repository and immutable commit;
- dependency lock or environment/image digest;
- dataset identifiers, immutable versions/checksums, and preprocessing version;
- configuration and hyperparameters;
- random seeds and hardware/runtime identity;
- metrics with definitions and evaluation dataset identity;
- checkpoints, logs, reports, and start/end/status timestamps.

Never commit raw secrets, restricted datasets, large generated artifacts, or
model weights merely to make them discoverable. Record governed locations and
immutable identifiers instead.

## GitLab experiment and model capabilities

When GitLab experiment tracking is selected, use its current MLflow-compatible
interface and verify supported client operations before adopting it. Treat
experiment artifacts stored through the package registry as governed packages:
define retention, access, cleanup, and promotion.

Use the model registry only after agreeing model naming, version immutability,
required metadata, validation evidence, lineage, owner, stages or aliases,
promotion approvals, rollback, and deployment consumption. A “best” run is not
promotable until its metric definition and validation contract are recorded.

## Durable workflow

Write experiment and model rules into agent-reachable project knowledge and
point to them from the entrypoint. Add a project-skill branch only when agents
repeatedly start, compare, or promote runs. Record which actions agents may do
unattended; model promotion, production deployment, data deletion, and access
changes remain human-approved by default.

Done when: a future agent can trace a deployed or registered model back to
code, environment, data, configuration, evaluation, and approval evidence, and
can reproduce the documented workflow without this builder.
