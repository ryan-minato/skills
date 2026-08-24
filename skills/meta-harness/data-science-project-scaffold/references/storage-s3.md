# S3 Storage

Use this branch when at least one source or product uses Amazon S3 or an
S3-compatible object store.

Merge the applicable source or product table from
[s3-config.toml](assets/storage/s3-config.toml) into
`config/project.toml`. Append a reworked
[s3-data-section.md](assets/storage/s3-data-section.md) for every S3
source to `.agents/knowledge/DATA.md`.

## Sources

- Record bucket, key, and object version ID. When versioning is unavailable,
  record ETag plus an independently verified checksum and document the
  weaker guarantee.
- Never use an unfixed prefix listing as the only source identity; persist
  the exact selected object keys and identities in run provenance.
- Stream or cache only outside `data/`. A project with no local source does
  not need `data/`.

## Products

- Give each workflow its own prefix and keep final products and provenance
  distinct.
- Write to a temporary key, validate it, then copy or promote it to the
  product key; delete the temporary object only after promotion succeeds.
- A project with no local product does not need `output/`.

Keep endpoint, bucket, prefix, and non-secret options in TOML. Put access
keys, session tokens, and private endpoints in environment variables.
Install only the S3 adapter the chosen processing stack needs; do not add
multiple clients as backups.

Verify current client configuration and object-version semantics in the
selected provider and adapter's official first-party documentation.
