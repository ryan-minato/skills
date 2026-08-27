# Data

Load before changing source acquisition, schemas, identities, storage
locations, transformations, or product publication.

## Contract

- Every source has a name, backend, URI, and immutable identity.
- Local original inputs live directly at `data/<source>/`; there is no
  `data/raw/`.
- `data/` never contains derived, cached, temporary, or final data.
- Only download workflows may publish a new local source path, and they refuse
  overwrite.
- Every product has an owner workflow and a distinct location.

## Source registry

| Source | Backend | Location | Immutable identity | Sensitivity | Structure |
|---|---|---|---|---|---|
| __SOURCE_NAME__ | __BACKEND__ | __URI__ | __VERSION_OR_CHECKSUM__ | __SENSITIVITY__ | __STRUCTURE__ |

## Product registry

| Product | Producer | Backend | Location | Consumers | Retention |
|---|---|---|---|---|---|
| __PRODUCT_NAME__ | __WORKFLOW__ | __BACKEND__ | __URI__ | __CONSUMERS__ | __RETENTION__ |

## Data-quality rules

__SCHEMA_INVARIANTS_AND_KNOWN_LIMITATIONS__

## Source-specific details

__APPEND_ONLY_THE_STORAGE_SECTIONS_FOR_BACKENDS_ACTUALLY_USED__
