# scaffold

[中文](README.zh.md)

Disposable, project-scoped builders for **projects of a specific topic** —
what such a project must contain, how its inputs, code, and outputs are
organized, and which conventions its agents inherit. Install **one** of them,
the one matching the project you are building; they are alternatives, not
layers. Deposit every lasting rule into the target project, verify it, then
remove the builders with `meta-disposal` from the `meta` catalog.

Generic harness machinery that stacks alongside any of these — complete harness
architecture, GitHub and GitLab lifecycle workflows, Python convention
defaults — lives in the disposable `meta` catalog. The two catalogs are
removed together by `meta-disposal` once the project is initialized.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [scaffold-colab](scaffold-colab/) | Scaffold a Colab-centric project whose deliverable is Google Colab notebooks: root-level notebooks mirrored to real Colab, an official Colab-runtime devcontainer for first-pass validation, colab-mcp connectivity, and a readable-notebook doctrine. |
| [scaffold-data-science](scaffold-data-science/) | Scaffold a reproducible Python data-science project: immutable raw inputs, a validated transformation pipeline, and data products that record where they came from, across local, S3, or Hugging Face storage. |
| [scaffold-ml](scaffold-ml/) | Scaffold one reproducible machine-learning project shape: a typed configuration surface with a resolved dump per run, a dependency carrier the user chooses (uv project by default, or uv-compiled requirements), an explicit Accelerate loop with a run manifest and a tracker, an evaluation entry bound to a recorded benchmark, an opt-in container recipe whose image digest is the environment identity, experiment-grade test, typing, and hook rules, a research-task convention, and agent guidance that points at the durable machine-learning skills by role. |
