# Skills

[中文](README.zh.md)

A curated library of [Agent Skills](https://agentskills.io) — self-contained
instruction packages that teach coding agents (Claude Code, Codex, Copilot,
and other compatible clients) how to perform specific tasks well.

## Catalogs

| Catalog | Contents | Install scope |
|---|---|---|
| [`core`](skills/core/) | Skills recommended for every environment | Global (user-level) |
| [`devcontainer`](skills/devcontainer/) | Dev Container authoring skills: Features, Templates, and prebuilt images | Per project, as needed |
| [`engineering`](skills/engineering/) | General programming methodology skills, plus GitHub/GitLab community authoring (templates, labels, conventions, health files) | Per project, as needed |
| [`meta-gitlab`](skills/meta-gitlab/) | GitLab harness authoring: agent tooling setup and project conventions (description templates, scoped labels, MR/commit/release rules, CI validation) | Per project, as needed |
| [`ops`](skills/ops/) | Platform operations: consolidated GitHub and GitLab day-to-day workflows with tooling setup | Per project, as needed |
| [`writing`](skills/writing/) | Human-audience writing: genre skills (academic, blog/opinion, copy) and medium skills (LaTeX, Typst, Markdown) | Per project, as needed |

Each catalog's README lists its skills.

## Installation

### Individual skills (skills CLI)

Install individual skills with the [skills CLI](https://github.com/vercel-labs/skills):

```bash
# Pick skills interactively (project-level)
npx skills add ryan-minato/skills

# Install a specific skill
npx skills add ryan-minato/skills --skill <skill-name>

# Install globally (recommended for core skills)
npx skills add ryan-minato/skills --skill <skill-name> -g
```

`core` skills are recommended for global installation so they are available
in every project; install other catalogs' skills into the projects that need
them.

### As Claude Code plugins

Each catalog is also published as its own Claude Code plugin. Add the
marketplace once, then install whole catalogs and enable only the ones you
need:

```
/plugin marketplace add ryan-minato/skills
/plugin install core@ryan-minato-skills
/plugin install ops@ryan-minato-skills      # or engineering@, writing@, ...
```

## Contributing

Conventions, quality standards, and repository mechanics are documented for
both humans and agents: start at [AGENTS.md](AGENTS.md), then
[ARCHITECTURE.md](ARCHITECTURE.md). Run `just setup` once after cloning and
`just check` before committing.

## License

[Apache-2.0](LICENSE)
