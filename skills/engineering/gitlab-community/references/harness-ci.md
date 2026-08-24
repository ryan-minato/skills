# GitLab harness CI

Read when a GitLab project needs a pipeline that mirrors its existing local
quality commands and gates merge requests.

1. Inspect the self-managed instance version when applicable, existing
   `.gitlab-ci.yml`, includes, workflow rules, runner tags, task runner,
   setup command, fast checks, lockfiles, permissions, and expected duration.
2. Run the local commands first. CI does not choose or repair the project's
   tests and linters.
3. Adapt [the pipeline shape](assets/harness-gitlab-ci.yml), replacing
   every placeholder and preserving the project's existing include structure.
4. Prefer tokenless jobs. Never expose protected variables to fork merge
   requests.
5. Read [ci-monorepo-paths.md](ci-monorepo-paths.md) for path-scoped jobs and
   [ci-slow-checks.md](ci-slow-checks.md) for work outside the normal MR
   feedback budget.
6. Avoid duplicate branch and merge-request pipelines; changing top-level
   workflow rules requires explicit agreement.
7. Record the selected jobs and required “Pipelines must succeed” activation
   in the durable harness, adapting
   [the AGENTS.md section](assets/agents-md-ci-section.md).

Done when: the pipeline mirrors passing local commands, works on the target
instance and runner, leaks no secret to forks, and its merge-gate activation is
recorded.
