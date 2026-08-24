# Hugging Face Storage

Use this branch when a source, product, or model lives on Hugging Face Hub.

Merge the applicable table from
[huggingface-config.toml](assets/storage/huggingface-config.toml) into
`config/project.toml`. Append a reworked
[huggingface-data-section.md](assets/storage/huggingface-data-section.md)
for every Hub source to `.agents/knowledge/DATA.md`.

## Sources and products

- Record repository ID, repository type, and immutable commit revision.
- Resolve a tag or branch to its commit before the run; never record
  `main`, another branch, or `latest` as the sole identity.
- Record the selected files and their revisions in provenance.
- Keep tokens in `.env`, never TOML, logs, notebooks, or reports.
- Omit local `data/` or `output/` when no local source or product needs it.

For a small project that stays entirely in the Hub data ecosystem, prefer
`datasets`. Do not add it when another chosen engine already reads and writes
the required Hub artifacts cleanly.

## Models

Record a model repository and immutable revision in configuration and run
provenance. Prefer Hub versioning over local weights; use the local-weight
branch only when the provider supplies no versioned repository.

Verify current revision resolution, upload, and download behavior in the
selected Hub client's official first-party documentation.
