## Guardrails

- Protected: <branches and tags the platform protects, and what it
  refuses — direct pushes, unapproved merges, failing checks>.
- Required checks: <names, verbatim>. These are produced by
  <workflow file>; renaming a job there updates this list and the
  platform settings in the same change.
- Ownership: `CODEOWNERS` routes review — <summary of who owns what>.
  New top-level paths get an owner line in the same PR that adds them.
- Dependency updates: <bot and cadence>; update PRs are handled by
  <who>, <merge policy>. An update that breaks CI is <policy>.
- Scanning: <secret/code scanning enabled, or the manual steps still
  pending and who executes them>.
