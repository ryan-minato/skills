## CI

CI mirrors the local checks: every job in
`.github/workflows/checks.yml` runs a command you can run yourself.

| CI job | Local command |
|---|---|
| <job name> | `<command>` |

- A CI failure reproduces locally with the mapped command; fix it there,
  never by editing the workflow to pass.
- Required checks: <which jobs block merging>. Renaming a job breaks the
  merge gate — update the repository's protection settings in the same
  change.
- Scheduled or manual jobs: <what runs outside PRs, and who watches it>.
