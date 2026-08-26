# Meta Skills

Disposable, project-scoped builders for creating a durable agent harness. Install
the catalog for a harness build, deposit every lasting rule into the target
project, verify it, then remove the builders with `meta-disposal`. Their shared
description prefix identifies them as temporary, allowing this catalog to give
project initialization complete guidance without carrying that context into
normal development.

| Skill | Purpose |
|---|---|
| [meta-harness-architecture](meta-harness-architecture/) | Investigate, plan, build, audit, and maintain a complete harness, including progressive loading, feedback loops, synchronization, and entropy management. |
| [meta-github-workflow](meta-github-workflow/) | Build or systematically repair a complete GitHub repository lifecycle harness for GitHub.com or GitHub Enterprise, designed around the pull-request loop: intake forms and Discussions routing, labels extending the defaults, tracking issues and milestones, early draft PRs via issue-linked branches, Actions quality gates and community automation, rulesets, CODEOWNERS, releases with label-driven notes, registries, optional Projects and ML experiment records, and durable project-agent workflows. |
| [meta-gitlab-workflow](meta-gitlab-workflow/) | Build or systematically repair a complete GitLab project lifecycle harness for gitlab.com or self-managed instances: planning, work items and early draft MRs, community files, CI/CD, governance, security, Wiki, releases, deployments, registries, optional MLOps, and durable project-agent workflows. |
| [meta-disposal](meta-disposal/) | Dry-run and remove copied disposable builders after fresh confirmation, without touching durable skills. |
| [python-project-defaults](python-project-defaults/) | Choose missing Python documentation, testing, and toolchain conventions without replacing working choices. |
| [ml-project-scaffold](ml-project-scaffold/) | Scaffold a quick ML experiment or maintainable training project with live GPU image discovery. |
| [data-science-project-scaffold](data-science-project-scaffold/) | Scaffold a reproducible Python data-science project with immutable inputs and product provenance. |
