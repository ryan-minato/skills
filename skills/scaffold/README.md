# scaffold

[中文](README.zh.md)

Disposable, project-scoped builders for **projects of a specific topic** —
what such a project must contain, how its inputs, code, and outputs are
organized, and which conventions its agents inherit. Install **one** of them,
the one matching the project you are building; they are alternatives, not
layers. Deposit every lasting rule into the target project, verify it, then
remove the builders with `scaffold-disposal`.

Generic harness machinery that stacks alongside any of these — complete harness
architecture, GitHub and GitLab lifecycle workflows, Python convention
defaults — lives in the disposable `meta` catalog. The two catalogs are
normally removed together once the project is initialized.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [scaffold-data-science](scaffold-data-science/) | Scaffold a reproducible Python data-science project: immutable raw inputs, a validated transformation pipeline, and data products that record where they came from, across local, S3, or Hugging Face storage. |
| [scaffold-disposal](scaffold-disposal/) | Dry-run and remove the copied disposable scaffold skills after fresh confirmation, without touching durable skills or another catalog's builders. |
| [scaffold-ml](scaffold-ml/) | Scaffold a machine-learning project as either a quick experiment or a maintainable training codebase, with hardware-aware dependencies, reproducible runs, and optional containers. |
