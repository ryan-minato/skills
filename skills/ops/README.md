# ops

[中文](README.zh.md)

**GitHub platform operations** — one consolidated playbook for day-to-day
forge work: issues, pull requests, CI, planning structures, releases,
Discussions, read-only repository research, guardrails, and the CLI/MCP
tooling setup that work depends on. It prefers the GitHub MCP server, falls
back to `gh`, provides an anonymous read-only REST tier, and embeds convention
discovery plus mandatory pre-publish review. Authoring GitHub templates,
labels, and policy belongs to `engineering`; building a complete GitLab
lifecycle harness belongs to the disposable `meta` catalog.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [github-ops](github-ops/) | Consolidated GitHub operations: issues, PRs, CI checks and Actions logs, Discussions, milestones/labels/Projects, releases, and read-only research on any repository — MCP-first with gh fallback, an anonymous REST read tier, template discovery before any create, and a mandatory pre-publish review; includes gh/MCP setup and five bundled scripts |
