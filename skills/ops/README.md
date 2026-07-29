# ops

[中文](README.zh.md)

**Platform operations** skills — one consolidated playbook per
collaboration platform for day-to-day forge work: issues, pull/merge
requests, CI, planning structures, releases, wikis, read-only repository
research, and the CLI/MCP tooling setup that work depends on. Each skill
keeps its platform's tool priority (GitHub: MCP server first, then `gh`;
GitLab: authenticated `glab` first, then the GitLab Duo MCP server), adds
an anonymous read-only REST fallback for when neither is available, and
embeds convention discovery plus a mandatory pre-publish review before
anything is published. Designed so small local models can execute them:
one recommended path per operation and decision tables throughout.
Authoring a project's templates, labels, and policies belongs to the
`engineering` catalog's community skills.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [github-ops](github-ops/) | Consolidated GitHub operations: issues, PRs, CI checks and Actions logs, Discussions, milestones/labels/Projects, releases, and read-only research on any repository — MCP-first with gh fallback, an anonymous REST read tier, template discovery before any create, and a mandatory pre-publish review; includes gh/MCP setup and five bundled scripts |
