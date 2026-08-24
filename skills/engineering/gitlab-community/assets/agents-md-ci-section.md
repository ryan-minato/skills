## CI

CI mirrors the local checks: every job in `.gitlab-ci.yml` runs a
command you can run yourself.

| CI job | Local command |
|---|---|
| <job name> | `<command>` |

- A CI failure reproduces locally with the mapped command; fix it there,
  never by editing the pipeline to pass.
- Blocking jobs: <which jobs the merge gate requires>. Renaming a job
  breaks the gate — update the project's merge settings in the same
  change.
- Runners: <who executes jobs — shared, instance, or project runners —
  and any constraints>.
- Scheduled or manual jobs: <what runs outside MRs, and who watches it>.
