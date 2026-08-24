# Architecture

## Purpose

__ARCHITECTURE_PURPOSE__

## Data flow

```text
__SOURCE_IDENTITIES__
        |
        v
src/__PACKAGE_NAME__/workflows/download_*   # acquisition only
        |
        v
configured immutable sources
        |
        v
src/__PACKAGE_NAME__/workflows/__PIPELINE_STEP__
        |
        v
configured step products -> final products + provenance -> report
```

## Modules

| Module | Responsibility | Must not do |
|---|---|---|
| `settings.py` | validate TOML and environment configuration | contain secrets or business logic |
| `sources/` | acquire and identify source versions | transform source data in place |
| `processing/` | reusable transformations and validations | choose storage from hidden globals |
| `workflows/` | thin, observable step entrypoints | hold reusable transformation logic |
| `data_guard.py` | snapshot and verify local original inputs | update a source baseline during a pipeline |

Delete absent modules and document every added production package.

## Storage

__STORAGE_ARCHITECTURE__

Each source and product declares its own backend and URI. A project may mix
local, S3, and Hugging Face locations. Local source data is always
`data/<source>/`; local products are always under `output/`.

## Reproducibility

Every run persists:

- exact source identities;
- resolved non-secret configuration;
- Git commit and `uv.lock` digest;
- model repository and immutable revision, when used;
- random seeds;
- step status and timing.

## Compute

__COMPUTE_DECISION_AND_SCALE_EVIDENCE__

## Model boundary

__MODEL_INFERENCE_BOUNDARY_OR_NONE__

Training, finetuning, optimizers, training loaders, and checkpoint management
are outside this repository.
