# Local Storage

Use this branch when at least one source or product uses the local
filesystem.

## Source data

For every local source, merge
[local-data-config.toml](assets/storage/local-data-config.toml) into
`config/project.toml`, append a reworked
[local-data-section.md](assets/storage/local-data-section.md) to
`.agents/knowledge/DATA.md`, and create `data/<source>/`.

`data/` is the original-input boundary:

- The first directory level identifies the source.
- A source may add a version directory when upstream publishes versions.
- Never create `data/raw/`.
- Never store transformed, cached, temporary, or final data in `data/`.
- Ignore `data/` in Git.

Put acquisition logic in `src/<package>/sources/<source>.py` and its thin
entry in `src/<package>/workflows/download_<source>.py`. The downloader must
write a temporary sibling, verify the expected checksum, and use an atomic
rename only when the final path does not exist. Refuse overwrite; a changed
upstream object gets a new version path.

Copy [local-data-guard.py](assets/storage/local-data-guard.py) to
`src/<package>/data_guard.py`. Snapshot immediately before a production run
and verify immediately after it. Keep the manifest with run provenance; a
remote product backend uploads the manifest with the product.

## Products

When any product is local, merge
[local-output-config.toml](assets/storage/local-output-config.toml) into
the config and create:

```text
output/
├── <step>/
├── final/
└── _provenance/
```

Each workflow writes only its own subtree. Write files to a temporary sibling
and atomically replace the derived destination after success. Derived output
may be regenerated; source data may not.

Verify current filesystem and adapter behavior in the selected implementation's
official first-party documentation.
