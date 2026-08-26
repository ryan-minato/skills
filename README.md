# Skills

[中文](README.zh.md)

A curated library of [Agent Skills](https://agentskills.io) — self-contained
instruction packages that teach coding agents (Claude Code, Codex, Copilot,
and other compatible clients) how to perform specific tasks well.

## Catalogs

| Catalog | Contents | Install scope |
|---|---|---|
| [`core`](skills/core/) | Skills recommended for every environment | Global (user-level) |
| [`engineering`](skills/engineering/) | General programming methodology skills, GitHub community authoring (templates, labels, conventions, health files), and narrow artifact-authoring workflows (e.g. Dev Container artifacts) | Per project, as needed |
| [`meta`](skills/meta/) | Disposable builders for complete agent harnesses, GitLab lifecycle workflows, DESIGN.md, and reproducible Python, ML, and data-science project scaffolds | Per project, remove after the harness is verified |
| [`ops`](skills/ops/) | GitHub day-to-day platform operations with tooling setup and pre-publish review | Per project, as needed |
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
in every project. Install other catalogs into the projects that need them;
`meta` contains temporary builders that should be removed after their
durable output is verified.

### As Claude Code plugins

Each catalog is also published as its own Claude Code plugin. Add the
marketplace once, then install whole catalogs and enable only the ones you
need:

```
/plugin marketplace add ryan-minato/skills
/plugin install core@ryan-minato-skills
/plugin install ops@ryan-minato-skills      # or engineering@, meta@, writing@, ...
```

## Contributing

Conventions, quality standards, and repository mechanics are documented for
both humans and agents: start at [AGENTS.md](AGENTS.md), then
[ARCHITECTURE.md](ARCHITECTURE.md). Run `just setup` once after cloning and
`just check` before committing.

## License

[Apache-2.0](LICENSE)
