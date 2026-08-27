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
