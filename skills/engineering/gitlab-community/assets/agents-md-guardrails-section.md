## Guardrails

- Protected: <branches and tags the platform protects, and what it
  refuses — direct pushes, unapproved merges, failing pipelines>.
- Merge gate: <approval rules and pipeline requirements, verbatim>.
  These key on jobs in `.gitlab-ci.yml`; renaming a job there updates
  the merge settings in the same change.
- Tier limits: <wanted rules the current tier cannot enforce — these are
  convention-only and reviewers watch for them>.
- Ownership: `CODEOWNERS` routes review — <summary of who owns what>.
  New top-level paths get an owner line in the same MR that adds them.
- Dependency updates: <bot and cadence>; update MRs are handled by
  <who>, <merge policy>. An update that breaks CI is <policy>.
- Scanning: <scanners enabled and where findings land, or the manual
  steps still pending and who executes them>.
